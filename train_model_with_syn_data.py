# train.py
# CPU-only Food-101 training (ResNet50) with auto-resume + history logging + optional MixUp
# - Train: Food-101 train split + ./synthetic_data/<class> images
# - Val:   pure Food-101 test split (no synthetic)
# - Tricks: CLAHE (train-only), label smoothing, cosine LR, weighted sampler
# - Auto-resume: loads model/optimizer/scheduler + RNG states, continues from last epoch
# - Logs metrics to CSV: ckpt_dir/training_log.csv

import os
import csv
import argparse
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.datasets import Food101
from tqdm import tqdm
from top_confusion_matrix import plot_top_confusions

import cv2
from PIL import Image, ImageEnhance, ImageFilter

# ----------------------------
# CPU / threading hygiene
# ----------------------------
torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
cv2.setNumThreads(0)
DEVICE = torch.device("cpu")  # Force CPU

# ----------------------------
# Defaults (CLI can override)
# ----------------------------
DEFAULT_DATA_ROOT = "./data"
DEFAULT_SYNTH_ROOT = "./synthetic_data"
DEFAULT_CKPT_DIR = "./checkpoints"
DEFAULT_LAST = "resnet50_food101_last.pth"
DEFAULT_BEST = "resnet50_food101_best.pth"
DEFAULT_LOG = "training_log.csv"
# Load class labels once at the start of your script
class_names = Food101(root="./data", download=False).classes

# Training recipe
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 30
DEFAULT_LR = 1e-4
DEFAULT_IMG_SIZE = 256
NUM_WORKERS = 0  # Windows/VM safe

# Pre/Post processing
APPLY_CLAHE = True            # train-only
APPLY_POSTPROCESS = False     # denoise is slow on CPU; keep off by default

# Regularization
LABEL_SMOOTH = 0.1


# ----------------------------
# Image processing helpers
# ----------------------------
def clahe_rgb(img_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def postprocess_pil(img_pil: Image.Image) -> Image.Image:
    img_pil = img_pil.filter(ImageFilter.SHARPEN)
    img_pil = ImageEnhance.Brightness(img_pil).enhance(1.05)
    return img_pil


# ----------------------------
# Train dataset merges synthetic images and exposes .samples/.targets
# ----------------------------
class Food101TrainWithSynthetic(Food101):
    """
    Food-101 train split + images from ./synthetic_data/<class>/
    Reconstructs a .samples list from internal fields, then appends synthetic files.
    Folder names in synthetic_data must match Food-101 class names exactly.
    """
    def __init__(self, *args, synthetic_root: str = None, **kwargs):
        super().__init__(*args, **kwargs)

        # class_to_idx from Food101's fixed class order
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Rebuild samples from Food101 internals (newer torchvision: _image_files/_labels)
        image_files = getattr(self, "_image_files", None)
        labels = getattr(self, "_labels", None)
        if image_files is None or labels is None:
            image_files = getattr(self, "imgs", None) or getattr(self, "images", None)
            labels = getattr(self, "labels", None)
        if image_files is None or labels is None:
            raise RuntimeError("Food101 internals changed—tell me your torchvision version.")

        self.samples: List[Tuple[str, int]] = [
            (os.path.abspath(p), int(y)) for p, y in zip(image_files, labels)
        ]

        # Append synthetic files if available
        if synthetic_root and os.path.isdir(synthetic_root):
            self.samples += self._load_synthetic_samples(synthetic_root)

        # Targets for sampler/metrics
        self.targets = [t for _, t in self.samples]

    def _load_synthetic_samples(self, root: str) -> List[Tuple[str, int]]:
        out: List[Tuple[str, int]] = []
        for cls_name in os.listdir(root):
            cls_dir = os.path.join(root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            label_idx = self.class_to_idx.get(cls_name)
            if label_idx is None:
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    out.append((os.path.abspath(os.path.join(cls_dir, fname)), label_idx))
        return out

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = cv2.imread(path)
        if img is None or img.size == 0:
            return self.__getitem__((index + 1) % len(self.samples))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if APPLY_CLAHE:
            img = clahe_rgb(img)
        img_pil = Image.fromarray(img)
        if APPLY_POSTPROCESS:
            img_pil = postprocess_pil(img_pil)
        if self.transform is not None:
            img_pil = self.transform(img_pil)
        return img_pil, target


# ----------------------------
# MixUp helper
# ----------------------------
def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float) -> tuple:
    """Returns mixed inputs, paired targets (y_a, y_b), and lambda."""
    if alpha <= 0:
        return x, (y, y), 1.0
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, (y_a, y_b), lam


# ----------------------------
# Checkpoint utils (with RNG states)
# ----------------------------
def save_checkpoint(path, epoch, model, optimizer, scheduler, best_val, args):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val": best_val,
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
        "args": vars(args),  # record hyperparams & mixup flags used
    }
    torch.save(state, path)

def load_checkpoint(path, model, optimizer, scheduler):
    # ckpt = torch.load(path, map_location=DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    # Restore RNG states for reproducibility
    if "rng_torch" in ckpt:
        torch.set_rng_state(ckpt["rng_torch"])
    if "rng_numpy" in ckpt:
        np.random.set_state(ckpt["rng_numpy"])
    if "rng_python" in ckpt:
        random.setstate(ckpt["rng_python"])
    epoch = ckpt.get("epoch", 0)
    best_val = ckpt.get("best_val", 0.0)
    prev_args = ckpt.get("args", {})
    return epoch, best_val, prev_args


# ----------------------------
# CSV logging
# ----------------------------
def init_csv_log(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "train_acc", "val_acc",
                "best_val", "mixup", "alpha", "batch_size", "lr", "img_size",
                "patience_counter", "early_stop_epoch"
            ])

def append_csv_log(csv_path, epoch, train_loss, train_acc, val_acc, best_val, args,patience_counter=None, early_stop_epoch=None):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, f"{train_loss:.6f}", f"{train_acc:.2f}", f"{val_acc:.2f}",
            f"{best_val:.2f}", int(args.mixup), args.alpha, args.batch_size, args.lr, args.img_size,
            patience_counter if patience_counter is not None else "",
            early_stop_epoch if early_stop_epoch is not None else ""
        ])


# ----------------------------
# Train
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train ResNet50 on Food-101 (CPU) with auto-resume and optional MixUp.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, type=str)
    parser.add_argument("--synthetic-root", default=DEFAULT_SYNTH_ROOT, type=str)
    parser.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR, type=str)
    parser.add_argument("--epochs", default=DEFAULT_EPOCHS, type=int)
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument("--img-size", default=DEFAULT_IMG_SIZE, type=int)
    parser.add_argument("--lr", default=DEFAULT_LR, type=float)
    parser.add_argument("--mixup", action="store_true", help="Enable MixUp regularization")
    parser.add_argument("--alpha", default=0.2, type=float, help="MixUp alpha")
    parser.add_argument("--no-resume", action="store_true", help="Do not auto-resume even if a checkpoint exists")
    args = parser.parse_args()

    data_root = args.data_root
    syn_root = args.synthetic_root
    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    last_path = os.path.join(ckpt_dir, DEFAULT_LAST)
    best_path = os.path.join(ckpt_dir, DEFAULT_BEST)
    log_path = os.path.join(ckpt_dir, DEFAULT_LOG)
    init_csv_log(log_path)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size + 32, args.img_size + 32)),
        transforms.RandomResizedCrop(args.img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Datasets
    train_ds = Food101TrainWithSynthetic(
        root=data_root, split="train", download=True,
        transform=train_transform, synthetic_root=syn_root
    )
    val_ds = Food101(root=data_root, split="test", download=False, transform=val_transform)

    # Weighted sampler for balance on merged train set
    class_counts = np.bincount(train_ds.targets, minlength=len(train_ds.classes))
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [float(class_weights[t]) for t in train_ds.targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # DataLoaders (CPU-friendly)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=False, persistent_workers=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False, persistent_workers=False
    )

    # Optional warmup to avoid first-iteration stall
    print("🔎 Warming up DataLoader…")
    _ = next(iter(train_loader))
    print("✅ Dataloader warmup ok")

    # Model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 101)
    model = model.to(DEVICE)

    # Loss / Optimizer / Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Auto-resume
    start_epoch, best_val = 0, 0.0
    if (not args.no_resume) and os.path.exists(last_path):
        print("🔁 Auto-resume: loading last checkpoint…")
        start_epoch, best_val, prev_args = load_checkpoint(last_path, model, optimizer, scheduler)
        print(f"✅ Resumed from epoch {start_epoch} (best_val={best_val:.2f}%)")
        # Optional: warn if major hyperparams changed since last run
        if prev_args:
            changed = []
            for k in ["batch_size", "img_size", "lr"]:
                old, new = prev_args.get(k), getattr(args, k)
                if old is not None and old != new:
                    changed.append(f"{k}: {old} -> {new}")
            if changed:
                print("⚠️ Hyperparameters changed since checkpoint:", "; ".join(changed))
    # Early stopping setup
    patience = 5  # stop after N epochs with no val improvement
    patience_counter = 0
    early_stop_triggered = False
    # Train loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        print(f"\n🔁 Epoch {epoch+1}/{args.epochs}  |  MixUp={'ON' if args.mixup else 'OFF'} (alpha={args.alpha if args.mixup else 0})")

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # MixUp (only if enabled)
            if args.mixup:
                inputs, (y_a, y_b), lam = mixup_batch(images, labels, args.alpha)
            else:
                inputs, (y_a, y_b), lam = images, (labels, labels), 1.0

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = lam * criterion(outputs, y_a) + (1.0 - lam) * criterion(outputs, y_b)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            # Training accuracy with hard labels (interpretable; lower with MixUp by design)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100.0 * correct / total if total else 0.0
        print(f"🟢 Train Loss: {run_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # Validation
        model.eval()
        v_correct, v_total = 0, 0
        y_true, y_pred = [], []   # collect for confusion matrix
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, pred = outputs.max(1)
                v_correct += pred.eq(labels).sum().item()
                v_total += labels.size(0)
                
                # collect predictions & ground truth
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
        val_acc = 100.0 * v_correct / v_total if v_total else 0.0
        print(f"🔵 Validation Accuracy: {val_acc:.2f}%")

        # Step LR
        scheduler.step()

        # Save checkpoints (last & best) + append CSV log
        save_checkpoint(last_path, epoch + 1, model, optimizer, scheduler, best_val, args)
        print(f"💾 Last checkpoint saved → {last_path}")

        if val_acc > best_val:
            best_val = val_acc
            patience_counter = 0  # reset if improvement
            save_checkpoint(best_path, epoch + 1, model, optimizer, scheduler, best_val, args)
            print(f"🏅 New best ({best_val:.2f}%) → {best_path}")
        else:
            patience_counter += 1
            print(f"⏳ No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                early_stop_triggered = True
                print("🛑 Early stopping triggered.")
                break

        # Append CSV log (now with patience info)
        append_csv_log(
            log_path, epoch + 1, run_loss, train_acc, val_acc, best_val, args,
            patience_counter=patience_counter,
            early_stop_epoch=(epoch+1 if early_stop_triggered else None)
        )
    print(f"\n✅ Training complete. Best validation accuracy: {best_val:.2f}%")
    plot_top_confusions(y_true, y_pred, class_names, top_n=10, save_path="top_confusion_matrix.png")
    print(f"   Best model: {best_path}")
    print(f"   Last model: {last_path}")
    print(f"   History:    {log_path}")


if __name__ == "__main__":
    main()