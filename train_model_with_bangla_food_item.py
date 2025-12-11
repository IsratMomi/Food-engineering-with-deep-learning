import os
import csv
import argparse
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset, ConcatDataset
from torch.optim.lr_scheduler import SequentialLR, LinearLR
from torchvision import models, transforms
from torchvision.datasets import Food101

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from tqdm import tqdm
import cv2
from PIL import Image, ImageEnhance, ImageFilter

from top_confusion_matrix import plot_top_confusions  # you already have this

# ------------------------------------------------------------------
# GLOBAL CONFIG
# ------------------------------------------------------------------
torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
cv2.setNumThreads(0)
DEVICE = torch.device("cpu")

# Adjust these paths for your environment
DEFAULT_DATA_ROOT = r"E:\er\project-food\data"  # contains food-101/
DEFAULT_FOOD_ROOT = os.path.join(DEFAULT_DATA_ROOT, "food-101")
DEFAULT_SYNTH_ROOT = os.path.join(DEFAULT_DATA_ROOT, "synthetic_data")

DEFAULT_BANGLA_ROOT = r"E:\er\project-food\data\bangla-food"  # has train/ and test/

DEFAULT_CKPT_DIR = "./checkpoints_cv_food_bangla"
DEFAULT_LOG = "training_log_all_folds.csv"
DEFAULT_SUMMARY = "training_summary.txt"
ENSEMBLE_CKPT = "resnet50_food101_ensemble.pth"
FINAL_120_CKPT = "resnet50_food101_bangla_120.pth"

DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 25
DEFAULT_LR = 1e-4
DEFAULT_IMG_SIZE = 256
NUM_WORKERS = 0

APPLY_CLAHE = True
APPLY_POSTPROCESS = True
LABEL_SMOOTH = 0.1
MIXUP_ALPHA_BASE = 0.2

N_FOLDS = 5
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ------------------------------------------------------------------
# IMAGE HELPERS
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# DATASET
# ------------------------------------------------------------------
class SimpleImageDataset(Dataset):
    """
    Dataset from explicit list of (abs_path, label).
    Can optionally apply CLAHE + postprocess before transforms.
    """

    def __init__(self,
                 samples: List[Tuple[str, int]],
                 transform=None,
                 apply_clahe: bool = False,
                 apply_postprocess: bool = False):
        self.samples = samples
        self.transform = transform
        self.apply_clahe = apply_clahe
        self.apply_postprocess = apply_postprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None or img.size == 0:
            # fall back to next sample if corrupted
            return self.__getitem__((idx + 1) % len(self.samples))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.apply_clahe:
            img = clahe_rgb(img)
        img_pil = Image.fromarray(img)

        if self.apply_postprocess:
            img_pil = postprocess_pil(img_pil)

        if self.transform is not None:
            img_pil = self.transform(img_pil)

        return img_pil, label


# ------------------------------------------------------------------
# MIXUP / CUTMIX
# ------------------------------------------------------------------
def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0:
        return x, (y, y), 1.0
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, (y_a, y_b), lam


def cutmix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    lam = np.random.beta(alpha, alpha)
    bsz, _, H, W = x.size()
    rand_index = torch.randperm(bsz)
    cx, cy = np.random.randint(W), np.random.randint(H)
    w = int(W * np.sqrt(1 - lam))
    h = int(H * np.sqrt(1 - lam))

    x1 = np.clip(cx - w // 2, 0, W)
    y1 = np.clip(cy - h // 2, 0, H)
    x2 = np.clip(cx + w // 2, 0, W)
    y2 = np.clip(cy + h // 2, 0, H)

    x[:, :, y1:y2, x1:x2] = x[rand_index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x, (y, y[rand_index]), lam


# ------------------------------------------------------------------
# MODEL + CHECKPOINT
# ------------------------------------------------------------------
def build_resnet(num_classes: int):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_val, extra: dict):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val": best_val,
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
        "extra": extra,
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if "rng_torch" in ckpt:
        torch.set_rng_state(ckpt["rng_torch"])
    if "rng_numpy" in ckpt:
        np.random.set_state(ckpt["rng_numpy"])
    if "rng_python" in ckpt:
        random.setstate(ckpt["rng_python"])
    return ckpt.get("epoch", 0), ckpt.get("best_val", 0.0), ckpt.get("extra", {})


# ------------------------------------------------------------------
# CSV LOGGING
# ------------------------------------------------------------------
def init_global_log(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "fold", "epoch",
                "train_loss", "train_acc",
                "val_acc", "best_val",
                "mixup", "alpha",
                "batch_size", "lr",
                "img_size"
            ])


def append_global_log(path, fold, epoch, train_loss, train_acc, val_acc,
                      best_val, mixup_on, alpha, batch_size, lr, img_size):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            fold, epoch,
            f"{train_loss:.6f}", f"{train_acc:.2f}",
            f"{val_acc:.2f}", f"{best_val:.2f}",
            int(mixup_on), alpha,
            batch_size, lr, img_size
        ])


# ------------------------------------------------------------------
# TRAIN / VAL LOOP
# ------------------------------------------------------------------
def train_one_epoch(model, train_loader, optimizer, criterion,
                    epoch, total_epochs, mixup_base_alpha, device):
    # Epoch-wise schedule
    if epoch < 1:
        current_mixup, current_alpha, use_erasing = False, 0.0, False
    elif epoch < int(0.6 * total_epochs):
        current_mixup, current_alpha, use_erasing = True, mixup_base_alpha, False
    elif epoch < int(0.8 * total_epochs):
        current_mixup, current_alpha, use_erasing = True, mixup_base_alpha / 2.0, True
    else:
        current_mixup, current_alpha, use_erasing = False, 0.0, True

    # Adjust RandomErasing in transform (if present)
    if hasattr(train_loader.dataset, "transform") and train_loader.dataset.transform is not None:
        tforms = train_loader.dataset.transform.transforms
        # remove existing RandomErasing
        tforms = [t for t in tforms if not isinstance(t, transforms.RandomErasing)]
        if use_erasing:
            tforms.append(
                transforms.RandomErasing(
                    p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value="random"
                )
            )
        train_loader.dataset.transform.transforms = tforms

    # Label smoothing schedule
    if isinstance(criterion, nn.CrossEntropyLoss):
        criterion.label_smoothing = 0.15 if current_mixup else 0.05

    model.train()
    run_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if current_mixup and current_alpha > 0:
            use_cutmix = random.random() < 0.5
            inputs, (y_a, y_b), lam = (
                cutmix_batch if use_cutmix else mixup_batch
            )(images, labels, current_alpha)
        else:
            inputs, y_a, y_b, lam = images, labels, labels, 1.0

        outputs = model(inputs)
        loss = lam * criterion(outputs, y_a) + (1.0 - lam) * criterion(outputs, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        run_loss += loss.item()
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

    train_loss = run_loss / max(1, len(train_loader))
    train_acc = 100.0 * correct / max(1, total)
    return train_loss, train_acc, current_mixup, current_alpha


def eval_model(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    acc = 100.0 * correct / max(1, total)
    return acc, y_true, y_pred


# ------------------------------------------------------------------
# FOOD-101 DATA LOADING
# ------------------------------------------------------------------
def load_food101_train_samples(root: str):
    ds = Food101(root=root, split="train", download=True)
    image_files = getattr(ds, "_image_files", None)
    labels = getattr(ds, "_labels", None)
    if image_files is None or labels is None:
        # fallback: use __getitem__
        image_files, labels = [], []
        for idx in range(len(ds)):
            img_path, target = ds._image_files[idx], ds._labels[idx]  # type: ignore
            image_files.append(img_path)
            labels.append(target)

    samples = [(os.path.abspath(p), int(y)) for p, y in zip(image_files, labels)]
    classes = ds.classes
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return samples, labels, classes, class_to_idx


def load_food101_test_samples(root: str):
    ds = Food101(root=root, split="test", download=True)
    image_files = getattr(ds, "_image_files", None)
    labels = getattr(ds, "_labels", None)
    samples = [(os.path.abspath(p), int(y)) for p, y in zip(image_files, labels)]
    return samples, labels


def load_synthetic_samples(syn_root: str, class_to_idx: dict) -> List[Tuple[str, int]]:
    if not os.path.isdir(syn_root):
        return []
    out: List[Tuple[str, int]] = []
    for cls_name in os.listdir(syn_root):
        cls_dir = os.path.join(syn_root, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        label_idx = class_to_idx.get(cls_name)
        if label_idx is None:
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                out.append((os.path.abspath(os.path.join(cls_dir, fname)), label_idx))
    return out


# ------------------------------------------------------------------
# BANGLA DATA LOADING
# ------------------------------------------------------------------
def collect_imagefolder_samples(root_dir: str, class_offset: int = 0):
    classes = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])
    cls_to_idx = {c: i + class_offset for i, c in enumerate(classes)}
    samples: List[Tuple[str, int]] = []
    for cls in classes:
        cdir = os.path.join(root_dir, cls)
        for fname in os.listdir(cdir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.abspath(os.path.join(cdir, fname)), cls_to_idx[cls]))
    return samples, classes, cls_to_idx


# ------------------------------------------------------------------
# AVERAGE CHECKPOINTS (ENSEMBLE)
# ------------------------------------------------------------------
def average_checkpoints(ckpt_paths: List[str], num_classes: int):
    model = build_resnet(num_classes)
    state_dicts = []
    for p in ckpt_paths:
        ckpt = torch.load(p, map_location=DEVICE)
        state_dicts.append(ckpt["model_state_dict"])

    avg_sd = {}
    keys = state_dicts[0].keys()
    for k in keys:
        tensors = [sd[k].float() for sd in state_dicts]
        avg_sd[k] = sum(tensors) / float(len(tensors))

    model.load_state_dict(avg_sd)
    return model, avg_sd


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, type=str)
    parser.add_argument("--food-root", default=DEFAULT_FOOD_ROOT, type=str)
    parser.add_argument("--synthetic-root", default=DEFAULT_SYNTH_ROOT, type=str)
    parser.add_argument("--bangla-root", default=DEFAULT_BANGLA_ROOT, type=str)
    parser.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR, type=str)
    parser.add_argument("--epochs", default=DEFAULT_EPOCHS, type=int)
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument("--img-size", default=DEFAULT_IMG_SIZE, type=int)
    parser.add_argument("--lr", default=DEFAULT_LR, type=float)
    parser.add_argument("--folds", default=N_FOLDS, type=int)
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    global_log_path = os.path.join(args.ckpt_dir, DEFAULT_LOG)
    init_global_log(global_log_path)

    # --------------------------------------------------------------
    # 1) LOAD FOOD-101 BASE + SYNTHETIC
    # --------------------------------------------------------------
    print("📦 Loading Food-101 train samples...")
    food_train_samples, food_train_labels, food_classes, food_class_to_idx = \
        load_food101_train_samples(args.food_root)
    num_food_classes = len(food_classes)
    print(f"Food-101 train images: {len(food_train_samples)} | classes: {num_food_classes}")

    print("📦 Loading Food-101 test samples...")
    food_test_samples, food_test_labels = load_food101_test_samples(args.food_root)
    print(f"Food-101 test images: {len(food_test_samples)}")

    print("🧩 Loading synthetic samples...")
    synth_samples = load_synthetic_samples(args.synthetic_root, food_class_to_idx)
    print(f"Synthetic images found: {len(synth_samples)}")

    # For dataloaders we need transforms
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size + 32, args.img_size + 32)),
        transforms.RandomResizedCrop(args.img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # --------------------------------------------------------------
    # 2) STRATIFIED K-FOLD ON FOOD-101 TRAIN (NO BANGLA HERE)
    # --------------------------------------------------------------
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    fold_accuracies = []
    best_ckpts = []

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(food_train_samples, food_train_labels), start=1):

        print(f"\n==================== FOLD {fold_idx}/{args.folds} ====================")
        fold_dir = os.path.join(args.ckpt_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        fold_best_path = os.path.join(fold_dir, f"resnet50_food101_fold{fold_idx}_best.pth")
        fold_last_path = os.path.join(fold_dir, f"resnet50_food101_fold{fold_idx}_last.pth")

        # Build fold-specific samples
        fold_train_samples = [food_train_samples[i] for i in train_idx] + synth_samples
        fold_train_labels = [food_train_labels[i] for i in train_idx] + \
                            [lbl for _, lbl in synth_samples]
        fold_val_samples = [food_train_samples[i] for i in val_idx]
        fold_val_labels = [food_train_labels[i] for i in val_idx]

        # Datasets
        train_ds = SimpleImageDataset(
            fold_train_samples,
            transform=train_transform,
            apply_clahe=APPLY_CLAHE,
            apply_postprocess=APPLY_POSTPROCESS
        )
        val_ds = SimpleImageDataset(
            fold_val_samples,
            transform=val_transform,
            apply_clahe=False,
            apply_postprocess=False
        )

        # Sampler for class balance
        class_counts = np.bincount(fold_train_labels, minlength=num_food_classes)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = [float(class_weights[t]) for t in fold_train_labels]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=NUM_WORKERS
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS
        )

        print("🔎 Dataloader warmup for this fold…")
        _ = next(iter(train_loader))
        print("✅ Fold dataloader warmup ok")

        # Model / criterion / optimizer / scheduler
        model = build_resnet(num_food_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        warmup_epochs = max(1, int(0.05 * args.epochs))
        main_epochs = args.epochs - warmup_epochs
        cosine_epochs = int(0.9 * main_epochs)
        cooldown_epochs = args.epochs - cosine_epochs - warmup_epochs

        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, cosine_epochs)
        )
        scheduler_cooldown = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, cooldown_epochs), eta_min=1e-6
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.3, total_iters=warmup_epochs),
                scheduler_cosine,
                scheduler_cooldown,
            ],
            milestones=[warmup_epochs, warmup_epochs + cosine_epochs],
        )

        best_val = 0.0

        for epoch in range(args.epochs):
            print(f"\n[Fold {fold_idx}] Epoch {epoch + 1}/{args.epochs}")
            train_loss, train_acc, mixup_on, cur_alpha = train_one_epoch(
                model, train_loader, optimizer, criterion,
                epoch, args.epochs, MIXUP_ALPHA_BASE, DEVICE
            )
            val_acc, y_true_fold, y_pred_fold = eval_model(
                model, val_loader, DEVICE
            )
            scheduler.step()

            lr_now = optimizer.param_groups[0]['lr']
            print(f"   Train Loss={train_loss:.4f} | "
                  f"Train Acc={train_acc:.2f}% | "
                  f"Val Acc={val_acc:.2f}% | LR={lr_now:.6e}")

            append_global_log(
                global_log_path, fold_idx, epoch + 1,
                train_loss, train_acc, val_acc,
                best_val, mixup_on, cur_alpha,
                args.batch_size, lr_now, args.img_size
            )

            # save last
            save_checkpoint(
                fold_last_path, epoch + 1, model, optimizer, scheduler,
                best_val, extra={"fold": fold_idx}
            )

            if val_acc > best_val:
                best_val = val_acc
                save_checkpoint(
                    fold_best_path, epoch + 1, model, optimizer, scheduler,
                    best_val, extra={"fold": fold_idx}
                )
                print(f"🏅 New best fold-{fold_idx} val: {best_val:.2f}%")

        print(f"✅ Fold {fold_idx} finished. Best val={best_val:.2f}%")
        fold_accuracies.append(best_val)
        best_ckpts.append(fold_best_path)

        # Final confusion / top-10 pairs for this fold
        try:
            # ensure using best weights
            load_checkpoint(fold_best_path, model)
            val_acc_final, y_true_final, y_pred_final = eval_model(model, val_loader, DEVICE)
            cm = confusion_matrix(y_true_final, y_pred_final, labels=list(range(num_food_classes)))
            cm_path = os.path.join(fold_dir, f"top10_confusions_fold{fold_idx}.png")
            plot_top_confusions(cm, food_classes, top_k=10, out_path=cm_path)
            print(f"📊 Saved top-10 confusion pairs for fold {fold_idx} → {cm_path}")
        except Exception as e:
            print(f"⚠️ Failed to save top confusion for fold {fold_idx}: {e}")

    # --------------------------------------------------------------
    # 3) ENSEMBLE OVER FOLDS
    # --------------------------------------------------------------
    print("\n==================== ENSEMBLE OVER FOLDS ====================")
    ensemble_model, ensemble_sd = average_checkpoints(best_ckpts, num_food_classes)
    ensemble_path = os.path.join(args.ckpt_dir, ENSEMBLE_CKPT)
    torch.save(
        {"model_state_dict": ensemble_sd, "fold_accuracies": fold_accuracies},
        ensemble_path
    )
    print(f"💾 Saved ensemble Food-101 model → {ensemble_path}")

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))
    print(f"Fold accuracies: {fold_accuracies}")
    print(f"Mean acc: {mean_acc:.2f}% | Std: {std_acc:.2f}%")

    # --------------------------------------------------------------
    # 4) B A N G L A   +  G L O B A L   F I N E - T U N E
    #    (No k-fold here, as requested)
    # --------------------------------------------------------------
    print("\n==================== B A N G L A  S T A G E ====================")
    bangla_train_root = os.path.join(args.bangla_root, "train")
    bangla_test_root = os.path.join(args.bangla_root, "test")

    if not (os.path.isdir(bangla_train_root) and os.path.isdir(bangla_test_root)):
        print(f"⚠️ Bangla root {args.bangla_root} missing train/ or test/. "
              f"Skipping Bangla integration.")
    else:
        bangla_train_samples, bangla_classes, _ = collect_imagefolder_samples(
            bangla_train_root, class_offset=num_food_classes
        )
        bangla_test_samples, _, _ = collect_imagefolder_samples(
            bangla_test_root, class_offset=num_food_classes
        )
        num_bangla = len(bangla_classes)
        num_total_classes = num_food_classes + num_bangla

        print(f"Bangla train images: {len(bangla_train_samples)} | "
              f"Bangla test images: {len(bangla_test_samples)} | "
              f"Bangla classes: {num_bangla}")

        # Build 120-class model by expanding ensemble model
        model_120 = build_resnet(num_total_classes).to(DEVICE)
        # Copy backbone + first 101 fc rows
        sd_120 = model_120.state_dict()
        for k, v in ensemble_sd.items():
            if k.startswith("fc."):
                continue
            if k in sd_120 and sd_120[k].shape == v.shape:
                sd_120[k].copy_(v)
        # FC: copy first 101 rows from ensemble
        fc_weight_old = ensemble_sd["fc.weight"]
        fc_bias_old = ensemble_sd["fc.bias"]
        sd_120["fc.weight"][:num_food_classes].copy_(fc_weight_old)
        sd_120["fc.bias"][:num_food_classes].copy_(fc_bias_old)
        model_120.load_state_dict(sd_120)

        # Merged train = Food-101 train + synthetic + Bangla train
        merged_train_samples = food_train_samples + synth_samples + bangla_train_samples
        merged_train_labels = food_train_labels + \
                              [lbl for _, lbl in synth_samples] + \
                              [lbl for _, lbl in bangla_train_samples]

        # Validation sets: Food-101 test + Bangla test (will evaluate separately)
        food_test_ds = SimpleImageDataset(
            food_test_samples, transform=val_transform,
            apply_clahe=False, apply_postprocess=False
        )
        bangla_test_ds = SimpleImageDataset(
            bangla_test_samples, transform=val_transform,
            apply_clahe=False, apply_postprocess=False
        )

        merged_train_ds = SimpleImageDataset(
            merged_train_samples,
            transform=train_transform,
            apply_clahe=APPLY_CLAHE,
            apply_postprocess=APPLY_POSTPROCESS
        )

        # Sampler
        merged_counts = np.bincount(merged_train_labels, minlength=num_total_classes)
        merged_weights = 1.0 / (merged_counts + 1e-6)
        merged_sample_weights = [float(merged_weights[t]) for t in merged_train_labels]
        merged_sampler = WeightedRandomSampler(
            merged_sample_weights,
            num_samples=len(merged_sample_weights),
            replacement=True
        )

        merged_train_loader = DataLoader(
            merged_train_ds,
            batch_size=args.batch_size,
            sampler=merged_sampler,
            num_workers=NUM_WORKERS
        )
        food_test_loader = DataLoader(
            food_test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS
        )
        bangla_test_loader = DataLoader(
            bangla_test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS
        )

        # Global fine-tune with small LR
        criterion_global = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
        optimizer_global = optim.Adam(model_120.parameters(), lr=args.lr / 10.0)

        warmup_g = max(1, int(0.1 * args.epochs))
        main_g = args.epochs - warmup_g
        cos_g = int(0.9 * main_g)
        cool_g = args.epochs - warmup_g - cos_g

        sched_cos_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_global, T_max=max(1, cos_g)
        )
        sched_cool_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_global, T_max=max(1, cool_g), eta_min=1e-6
        )
        scheduler_global = SequentialLR(
            optimizer_global,
            schedulers=[
                LinearLR(optimizer_global, start_factor=0.3, total_iters=warmup_g),
                sched_cos_g,
                sched_cool_g,
            ],
            milestones=[warmup_g, warmup_g + cos_g],
        )

        best_global_food = 0.0
        best_global_bangla = 0.0

        for epoch in range(args.epochs):
            print(f"\n[GLOBAL Food+Bangla] Epoch {epoch + 1}/{args.epochs}")
            g_train_loss, g_train_acc, mixup_on_g, alpha_g = train_one_epoch(
                model_120, merged_train_loader, optimizer_global,
                criterion_global, epoch, args.epochs,
                MIXUP_ALPHA_BASE, DEVICE
            )

            food_acc, _, _ = eval_model(model_120, food_test_loader, DEVICE)
            bangla_acc, _, _ = eval_model(model_120, bangla_test_loader, DEVICE)
            scheduler_global.step()

            lr_g = optimizer_global.param_groups[0]['lr']
            print(f"   Train Loss={g_train_loss:.4f} | Train Acc={g_train_acc:.2f}% "
                  f"| Food101 Test Acc={food_acc:.2f}% "
                  f"| Bangla Test Acc={bangla_acc:.2f}% | LR={lr_g:.6e}")

            if food_acc > best_global_food:
                best_global_food = food_acc
            if bangla_acc > best_global_bangla:
                best_global_bangla = bangla_acc

        final_120_path = os.path.join(args.ckpt_dir, FINAL_120_CKPT)
        torch.save(
            {"model_state_dict": model_120.state_dict(),
             "best_food_acc": best_global_food,
             "best_bangla_acc": best_global_bangla,
             "num_food_classes": num_food_classes,
             "num_bangla_classes": num_bangla},
            final_120_path
        )
        print(f"💾 Saved final 120-class Food+Bangla model → {final_120_path}")
        print(f"Best Food-101 test acc: {best_global_food:.2f}%")
        print(f"Best Bangla test acc: {best_global_bangla:.2f}%")

    # --------------------------------------------------------------
    # 5) SUMMARY REPORT
    # --------------------------------------------------------------
    summary_path = os.path.join(args.ckpt_dir, DEFAULT_SUMMARY)
    with open(summary_path, "w") as f:
        f.write("Food-101 5-Fold CV Results\n")
        f.write("===========================\n")
        for i, acc in enumerate(fold_accuracies, start=1):
            f.write(f"Fold {i}: {acc:.2f}%\n")
        f.write(f"\nMean accuracy: {mean_acc:.2f}%\n")
        f.write(f"Std accuracy : {std_acc:.2f}%\n")
        f.write(f"\nEnsemble checkpoint: {ensemble_path}\n")
        f.write(f"Global ckpt dir: {args.ckpt_dir}\n")

    print(f"\n📝 Summary report written to: {summary_path}")
    print("✅ Done.")


if __name__ == "__main__":
    main()