import os
import sys
import csv
import random
import math
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from torch.optim.lr_scheduler import SequentialLR, LinearLR
from torchvision import transforms, models
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# -----------------------
# ============== CONFIG =
# -----------------------
DEFAULT_FOOD101_ROOT = r"E:\er\project-food\data\food-101"
DEFAULT_BANGLA_ROOT = r"E:\er\project-food\data\bangla-food"
DEFAULT_SYNTH_ROOT = r"E:\er\project-food\data\synthetic_data"
DEFAULT_CKPT_DIR = r"./checkpoints"
DEFAULT_LOG = "training_log.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_BATCH = 32
DEFAULT_IMG = 256
DEFAULT_EPOCHS_FOOD = 20
DEFAULT_EPOCHS_BANGLA_HEAD = 15
DEFAULT_EPOCHS_GLOBAL = 10

DEFAULT_LR_FOOD = 1e-3
DEFAULT_LR_BANGLA = 1e-3
DEFAULT_LR_GLOBAL = 1e-4
DEFAULT_MIXUP_ALPHA = 0.4

NUM_WORKERS = 0
LABEL_SMOOTH = 0.1
APPLY_CLAHE = True
APPLY_POSTPROCESS = True

FORCE_TRAIN_FOOD = False
UNFREEZE_LAYER4 = True
UNFREEZE_BLOCKS = 1

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(DEFAULT_CKPT_DIR, exist_ok=True)

# -----------------------
# ============== HELPERS =
# -----------------------
def clahe_pil(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb)

def postprocess_pil(img: Image.Image) -> Image.Image:
    img = img.filter(ImageFilter.SHARPEN)
    img = ImageEnhance.Brightness(img).enhance(1.05)
    return img

def save_confusion_matrix(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.title(f'Confusion matrix ({len(labels)} classes)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"🖼️ Saved confusion matrix → {out_path}")

def init_csv(path):
    if not os.path.exists(path):
        with open(path,'w',newline='') as f:
            w = csv.writer(f)
            w.writerow(["timestamp","stage","epoch","train_loss","train_acc","val_loss","val_acc","lr"])

def append_csv(path, row):
    with open(path,'a',newline='') as f:
        w = csv.writer(f)
        w.writerow(row)

# -----------------------------------------------------
# 🔥🔥🔥 MISSING FUNCTION ADDED — NOW FIXED 🔥🔥🔥
# -----------------------------------------------------
def save_state(path, epoch, model, optimizer, scheduler, best_val, extra_args=None):
    """Unified checkpoint saving function."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_val": best_val,
        "extra": extra_args,
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
    }
    torch.save(state, path)
    print(f"💾 Saved checkpoint → {path}")
# -----------------------------------------------------

# -----------------------
# ============== DATASETS =
# -----------------------
class FilesDataset(Dataset):
    def __init__(self, samples: List[Tuple[str,int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

def make_imagefolder_samples(root_dir: str):
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,d))])
    cls_to_idx = {c:i for i,c in enumerate(classes)}
    samples = []
    for c in classes:
        cdir = os.path.join(root_dir, c)
        for fname in os.listdir(cdir):
            if fname.lower().endswith((".jpg",".jpeg",".png")):
                samples.append((os.path.join(cdir,fname), cls_to_idx[c]))
    return samples, classes

def read_food101_meta_test(food101_root: str, images_root: str):
    meta_dir = os.path.join(food101_root, "meta")
    test_file = os.path.join(meta_dir, "test.txt")
    classes_file = os.path.join(meta_dir, "classes.txt")
    if not os.path.exists(test_file) or not os.path.exists(classes_file):
        return []
    with open(classes_file, "r", encoding="utf8") as f:
        classes = [l.strip() for l in f.readlines() if l.strip()]
    cls_to_idx = {c:i for i,c in enumerate(classes)}
    samples = []
    with open(test_file, "r", encoding="utf8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            cls_name = ln.split("/")[0]
            fname = "/".join(ln.split("/")[1:])
            full = os.path.join(images_root, cls_name, fname)
            if os.path.exists(full):
                samples.append((full, cls_to_idx[cls_name]))
    return samples

# -----------------------
# ============== MODEL =
# -----------------------
def build_resnet(num_classes:int, pretrained=True):
    if pretrained:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# -----------------------
# ============== TRAIN/VAL =
# -----------------------
def mixup_batch(x, y, alpha):
    if alpha <= 0:
        return x, (y, y), 1.0
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    return mixed_x, (y, y[index]), lam

def cutmix_batch(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = x.size()
    rand_index = torch.randperm(B)
    cx, cy = np.random.randint(W), np.random.randint(H)
    w = int(W * math.sqrt(1 - lam))
    h = int(H * math.sqrt(1 - lam))
    x1, y1 = max(cx - w//2, 0), max(cy - h//2, 0)
    x2, y2 = min(cx + w//2, W), min(cy + h//2, H)
    x[:, :, y1:y2, x1:x2] = x[rand_index, :, y1:y2, x1:x2]
    lam = 1 - ((x2-x1)*(y2-y1)/(W*H))
    return x, (y, y[rand_index]), lam

def train_epoch(model, loader, optimizer, criterion, device, mixup_alpha=0.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="Training", ncols=100):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if mixup_alpha > 0:
            if random.random() < 0.5:
                inputs, (y_a, y_b), lam = cutmix_batch(imgs.clone(), labels, mixup_alpha)
            else:
                inputs, (y_a, y_b), lam = mixup_batch(imgs.clone(), labels, mixup_alpha)
            outputs = model(inputs)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total

def validate_epoch(model, loader, criterion, device, collect_preds=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_true, all_pred = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validation", ncols=100):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            _, preds = outputs.max(1)

            total_loss += loss.item() * labels.size(0)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            if collect_preds:
                all_true.extend(labels.cpu().tolist())
                all_pred.extend(preds.cpu().tolist())

    if collect_preds:
        return total_loss / total, 100.0 * correct / total, all_true, all_pred
    return total_loss / total, 100.0 * correct / total, None, None

# -----------------------
# ============== MAIN =
# -----------------------
def main():
    print(f"Device: {DEVICE}")
    csv_log = os.path.join(DEFAULT_CKPT_DIR, DEFAULT_LOG)
    init_csv(csv_log)
    cand1 = os.path.join(DEFAULT_FOOD101_ROOT, "images")
    cand2 = os.path.join(DEFAULT_FOOD101_ROOT, "train")
    if os.path.isdir(cand1):
        food_images_root = cand1
    elif os.path.isdir(cand2):
        food_images_root = cand2
    else:
        print("ERROR: Could not find Food-101 images under DEFAULT_FOOD101_ROOT. Exiting.")
        sys.exit(1)

    food_classes = sorted([d for d in os.listdir(food_images_root)
                           if os.path.isdir(os.path.join(food_images_root, d))])
    num_food = len(food_classes)
    print(f"Food101 classes found: {num_food} (example: {food_classes[:3]})")

    # 2) Bangla dataset
    bangla_train_root = os.path.join(DEFAULT_BANGLA_ROOT, "train")
    bangla_test_root = os.path.join(DEFAULT_BANGLA_ROOT, "test")
    if not os.path.isdir(bangla_train_root) or not os.path.isdir(bangla_test_root):
        print("ERROR: Bangla dataset requires 'train' and 'test' subfolders.")
        sys.exit(1)

    bangla_classes = sorted([d for d in os.listdir(bangla_train_root)
                             if os.path.isdir(os.path.join(bangla_train_root, d))])
    num_bangla = len(bangla_classes)
    print(f"Bangla classes found: {num_bangla} (example: {bangla_classes[:3]})")

    num_total = num_food + num_bangla
    print(f"Total unified classes = {num_total}")

    # -----------------------
    # Transformations
    # -----------------------
    train_tf = transforms.Compose([
        transforms.Lambda(lambda img: clahe_pil(img) if APPLY_CLAHE else img),
        transforms.RandomResizedCrop(224, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    print("\n📥 Loading Food-101 train/val ...")
    food_train_samples, _ = make_imagefolder_samples(food_images_root)

    food_meta_test = read_food101_meta_test(DEFAULT_FOOD101_ROOT, food_images_root)
    if food_meta_test:
        print(f"✅ Using official Food-101 test split → {len(food_meta_test)} images")
        food_val_samples = food_meta_test
    else:
        print("⚠️ test.txt missing → Using training split as validation (not ideal)")
        food_val_samples = food_train_samples

    food_train_ds = FilesDataset(food_train_samples, transform=train_tf)
    food_val_ds = FilesDataset(food_val_samples, transform=val_tf)

    # -----------------------
    # Synthetic data merge
    # -----------------------
    synth_root = DEFAULT_SYNTH_ROOT
    if os.path.isdir(synth_root):
        print("🧩 Merging synthetic samples...")
        synth_samples = []
        for cls in food_classes:
            cdir = os.path.join(synth_root, cls)
            if not os.path.isdir(cdir):
                continue
            for fname in os.listdir(cdir):
                if fname.lower().endswith((".jpg",".jpeg",".png")):
                    synth_samples.append((os.path.join(cdir,fname), food_classes.index(cls)))

        if synth_samples:
            synth_ds = FilesDataset(synth_samples, transform=train_tf)
            food_train_ds = ConcatDataset([synth_ds, food_train_ds])
            print(f"🧩 Added {len(synth_samples)} synthetic images.")
        else:
            print("⚠️ No synthetic images found.")
    else:
        print("ℹ️ No synthetic folder present.")

    print(f"📂 Food train samples: {len(food_train_ds)} | Food val samples: {len(food_val_ds)}")

    # -----------------------
    # Bangla Dataset Loading
    # -----------------------
    print("\n📥 Loading Bangla train/test ...")
    bangla_train_samples, _ = make_imagefolder_samples(bangla_train_root)
    bangla_test_samples, _ = make_imagefolder_samples(bangla_test_root)

    bangla_train_samples = [(p, idx + num_food) for (p, idx) in bangla_train_samples]
    bangla_test_samples = [(p, idx + num_food) for (p, idx) in bangla_test_samples]

    bangla_train_ds = FilesDataset(bangla_train_samples, transform=train_tf)
    bangla_val_ds = FilesDataset(bangla_test_samples, transform=val_tf)

    print(f"📦 Bangla train samples: {len(bangla_train_ds)}")

    # -----------------------
    # Stage 1 — Food-101 Training
    # -----------------------
    food_best_ckpt = os.path.join(DEFAULT_CKPT_DIR, "food_best.pth")
    model_food = build_resnet(num_food, pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    do_train_food = FORCE_TRAIN_FOOD or (not os.path.exists(food_best_ckpt))

    if not do_train_food:
        print("🔁 Loading existing Food-101 model…")
        try:
            state = torch.load(food_best_ckpt, map_location=DEVICE)
            model_food.load_state_dict(state["model_state_dict"])
            best_food_val = state.get("best_val", 0.0)
            print(f"✅ Loaded best Food-101 model (val={best_food_val:.2f}%)")
        except Exception as e:
            print("❌ Failed to load checkpoint:", e)
            do_train_food = True

    if do_train_food:
        print("\n========== STAGE 1: Training Food-101 ==========")

        # Collect labels for sampler
        def get_labels(dataset):
            labels = []
            if isinstance(dataset, ConcatDataset):
                for ds in dataset.datasets:
                    labels.extend([lbl for _, lbl in ds.samples])
            else:
                labels.extend([lbl for _, lbl in dataset.samples])
            return labels

        food_labels = get_labels(food_train_ds)
        counts = np.bincount(food_labels, minlength=num_food)
        class_w = 1.0 / np.maximum(counts, 1)
        sample_weights = [class_w[l] for l in food_labels]

        sampler = WeightedRandomSampler(sample_weights,
                                        num_samples=len(sample_weights),
                                        replacement=True)

        food_loader = DataLoader(food_train_ds,
                                 batch_size=DEFAULT_BATCH,
                                 sampler=sampler,
                                 num_workers=NUM_WORKERS)

        food_val_loader = DataLoader(food_val_ds,
                                     batch_size=DEFAULT_BATCH,
                                     shuffle=False,
                                     num_workers=NUM_WORKERS)

        optimizer_food = optim.Adam(model_food.parameters(), lr=DEFAULT_LR_FOOD)

        warmup = max(1, int(0.05 * DEFAULT_EPOCHS_FOOD))
        main_epochs = DEFAULT_EPOCHS_FOOD - warmup
        cosine_epochs = int(0.9 * main_epochs)
        cooldown_epochs = DEFAULT_EPOCHS_FOOD - warmup - cosine_epochs

        scheduler_food = SequentialLR(
            optimizer_food,
            schedulers=[
                LinearLR(optimizer_food, start_factor=0.3, total_iters=warmup),
                optim.lr_scheduler.CosineAnnealingLR(optimizer_food, T_max=cosine_epochs),
                optim.lr_scheduler.CosineAnnealingLR(optimizer_food,
                                                     T_max=cooldown_epochs, eta_min=1e-6),
            ],
            milestones=[warmup, warmup + cosine_epochs]
        )

        best_food_val = 0.0

        for epoch in range(DEFAULT_EPOCHS_FOOD):
            # mixup schedule
            if epoch < 1:
                mixup_alpha = 0
            elif epoch < int(0.6 * DEFAULT_EPOCHS_FOOD):
                mixup_alpha = DEFAULT_MIXUP_ALPHA
            elif epoch < int(0.8 * DEFAULT_EPOCHS_FOOD):
                mixup_alpha = DEFAULT_MIXUP_ALPHA / 2
            else:
                mixup_alpha = 0

            print(f"\n[Food101] Epoch {epoch+1}/{DEFAULT_EPOCHS_FOOD} | MixUp α={mixup_alpha}")

            tr_loss, tr_acc = train_epoch(model_food, food_loader, optimizer_food,
                                          criterion, DEVICE, mixup_alpha)

            val_loss, val_acc, _, _ = validate_epoch(model_food, food_val_loader,
                                                     criterion, DEVICE)

            scheduler_food.step()

            lr_now = optimizer_food.param_groups[0]['lr']
            append_csv(csv_log, [
                datetime.now().isoformat(), "food", epoch+1,
                tr_loss, tr_acc, val_loss, val_acc, lr_now
            ])

            print(f"[Food101] TrainAcc={tr_acc:.2f}% | ValAcc={val_acc:.2f}%")

            save_state(os.path.join(DEFAULT_CKPT_DIR, "food_last.pth"),
                       epoch+1, model_food, optimizer_food, scheduler_food,
                       best_food_val, {"stage": "food"})

            if val_acc > best_food_val:
                best_food_val = val_acc
                save_state(food_best_ckpt, epoch+1, model_food,
                           optimizer_food, scheduler_food,
                           best_food_val, {"stage": "food"})
                print(f"🏅 New Best Food-101 val={best_food_val:.2f}%")

        print(f"✅ Finished Food-101 Training (best={best_food_val:.2f}%)")

    # ----------------------------
    # Optional confusion matrix for Food101
    # ----------------------------
    try:
        if os.path.exists(food_best_ckpt):
            print("Generating Food-101 confusion matrix...")
            food_val_loader = DataLoader(food_val_ds, batch_size=DEFAULT_BATCH,
                                         shuffle=False, num_workers=NUM_WORKERS)
            _, _, y_true_food, y_pred_food = validate_epoch(
                model_food, food_val_loader, criterion, DEVICE, collect_preds=True
            )
            save_confusion_matrix(
                y_true_food, y_pred_food, food_classes,
                os.path.join(DEFAULT_CKPT_DIR, "confusion_food101.png")
            )
    except Exception as e:
        print("⚠️ Failed to generate Food-101 confusion matrix:", e)

    # ----------------------------
    # Stage 2 — Expand classifier for unified classes
    # ----------------------------
    print("\n========== STAGE 2: Expanding model for unified classes ==========")

    model_unified = build_resnet(num_total, pretrained=False).to(DEVICE)

    sd_food = model_food.state_dict()
    sd_unified = model_unified.state_dict()

    copied = 0
    for k, v in sd_food.items():
        if k.startswith("fc."):
            continue
        if k in sd_unified and sd_unified[k].shape == v.shape:
            sd_unified[k].copy_(v)
            copied += 1

    model_unified.load_state_dict(sd_unified)
    print(f"🔁 Copied {copied} backbone parameters from Food-101 model.")

    # ----------------------------
    # Stage 3 — Bangla head-only training
    # ----------------------------
    print("\n========== STAGE 3: Bangla head-only training ==========")

    for name, p in model_unified.named_parameters():
        p.requires_grad = name.startswith("fc")

    optimizer_bh = optim.Adam(
        filter(lambda p: p.requires_grad, model_unified.parameters()),
        lr=DEFAULT_LR_BANGLA
    )
    scheduler_bh = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_bh, T_max=max(1, DEFAULT_EPOCHS_BANGLA_HEAD)
    )

    bangla_train_loader = DataLoader(
        bangla_train_ds, batch_size=DEFAULT_BATCH,
        shuffle=True, num_workers=NUM_WORKERS
    )
    bangla_val_loader = DataLoader(
        bangla_val_ds, batch_size=DEFAULT_BATCH,
        shuffle=False, num_workers=NUM_WORKERS
    )

    best_bangla = 0.0

    for epoch in range(DEFAULT_EPOCHS_BANGLA_HEAD):
        print(f"\n[Bangla Head] Epoch {epoch+1}/{DEFAULT_EPOCHS_BANGLA_HEAD}")

        tr_loss, tr_acc = train_epoch(
            model_unified, bangla_train_loader,
            optimizer_bh, criterion, DEVICE, mixup_alpha=0
        )

        val_loss, val_acc, y_true_b, y_pred_b = validate_epoch(
            model_unified, bangla_val_loader,
            criterion, DEVICE, collect_preds=True
        )

        scheduler_bh.step()

        lr_now = optimizer_bh.param_groups[0]['lr']
        append_csv(csv_log, [
            datetime.now().isoformat(), "bangla_head", epoch+1,
            tr_loss, tr_acc, val_loss, val_acc, lr_now
        ])

        print(f"[Bangla Head] TrainAcc={tr_acc:.2f}% | ValAcc={val_acc:.2f}%")

        save_state(os.path.join(DEFAULT_CKPT_DIR, "bangla_head_last.pth"),
                   epoch+1, model_unified, optimizer_bh, scheduler_bh,
                   best_bangla, {"stage": "bangla_head"})

        if val_acc > best_bangla:
            best_bangla = val_acc
            save_state(os.path.join(DEFAULT_CKPT_DIR, "bangla_head_best.pth"),
                       epoch+1, model_unified, optimizer_bh, scheduler_bh,
                       best_bangla, {"stage":"bangla_head"})
            print(f"🏅 New Best Bangla Validation={best_bangla:.2f}%")

    # Store confusion matrix for Bangla
    try:
        if os.path.exists(os.path.join(DEFAULT_CKPT_DIR, "bangla_head_best.pth")):
            labels_bangla = [c for c in bangla_classes]
            save_confusion_matrix(
                y_true_b, y_pred_b, list(range(num_food, num_food + num_bangla)),
                os.path.join(DEFAULT_CKPT_DIR, "confusion_bangla_head.png")
            )
    except Exception as e:
        print("⚠️ Failed Bangla confusion matrix:", e)

    print(f"✅ Finished Bangla head-only training (best={best_bangla:.2f}%)")

    # ----------------------------
    # Stage 4 — Global partial unfreeze training
    # ----------------------------
    print("\n========== STAGE 4: Global partial unfreeze ==========")

    for name, param in model_unified.named_parameters():
        if name.startswith("layer4") or name.startswith("fc"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    print("✔ Unfrozen layers: layer4 + fc")

    merged_train = ConcatDataset([food_train_ds, bangla_train_ds])
    merged_val = ConcatDataset([food_val_ds, bangla_val_ds])

    merged_labels = []
    for sub in merged_train.datasets:
        merged_labels.extend([lbl for _, lbl in sub.samples])

    merged_counts = np.bincount(merged_labels, minlength=num_total)
    merged_w = 1.0 / np.maximum(merged_counts, 1)
    merged_weights = [merged_w[y] for y in merged_labels]

    merged_sampler = WeightedRandomSampler(
        merged_weights,
        num_samples=len(merged_weights),
        replacement=True
    )

    merged_loader = DataLoader(
        merged_train, batch_size=DEFAULT_BATCH,
        sampler=merged_sampler, num_workers=NUM_WORKERS
    )

    merged_val_loader = DataLoader(
        merged_val, batch_size=DEFAULT_BATCH,
        shuffle=False, num_workers=NUM_WORKERS
    )

    optimizer_glob = optim.Adam(
        filter(lambda p: p.requires_grad, model_unified.parameters()),
        lr=DEFAULT_LR_GLOBAL
    )
    scheduler_glob = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_glob, T_max=max(1, DEFAULT_EPOCHS_GLOBAL)
    )

    best_global_val = 0.0
    y_true_g, y_pred_g = [], []

    for epoch in range(DEFAULT_EPOCHS_GLOBAL):
        print(f"\n[GLOBAL] Epoch {epoch+1}/{DEFAULT_EPOCHS_GLOBAL}")

        tr_loss, tr_acc = train_epoch(
            model_unified, merged_loader,
            optimizer_glob, criterion, DEVICE,
            mixup_alpha=DEFAULT_MIXUP_ALPHA
        )

        val_loss, val_acc, y_true_g, y_pred_g = validate_epoch(
            model_unified, merged_val_loader,
            criterion, DEVICE, collect_preds=True
        )

        scheduler_glob.step()

        lr_now = optimizer_glob.param_groups[0]['lr']
        append_csv(csv_log, [
            datetime.now().isoformat(), "global", epoch+1,
            tr_loss, tr_acc, val_loss, val_acc, lr_now
        ])

        print(f"[GLOBAL] TrainAcc={tr_acc:.2f}% | ValAcc={val_acc:.2f}%")

        save_state(os.path.join(DEFAULT_CKPT_DIR, "global_last.pth"),
                   epoch+1, model_unified, optimizer_glob, scheduler_glob,
                   best_global_val, {"stage": "global"})

        if val_acc > best_global_val:
            best_global_val = val_acc
            save_state(os.path.join(DEFAULT_CKPT_DIR, "global_best.pth"),
                       epoch+1, model_unified, optimizer_glob, scheduler_glob,
                       best_global_val, {"stage": "global"})
            print(f"🏅 New BEST Global val={best_global_val:.2f}%")

    # ----------------------------
    # Global confusion matrix
    # ----------------------------
    try:
        if len(y_true_g) > 0:
            all_labels = food_classes + bangla_classes
            save_confusion_matrix(
                y_true_g, y_pred_g, all_labels,
                os.path.join(DEFAULT_CKPT_DIR, "confusion_global_120.png")
            )
    except Exception as e:
        print("⚠️ Failed global confusion matrix:", e)

    # ----------------------------
    # Summary
    # ----------------------------
    print("\n✅ Training pipeline complete.")
    try:
        print(f"Best Food101 val: {best_food_val:.2f}%")
    except:
        print("Best Food101 val: (skipped)")

    print(f"Best Bangla val: {best_bangla:.2f}%")
    print(f"Best Global val: {best_global_val:.2f}%")
    print(f"Checkpoints saved in: {DEFAULT_CKPT_DIR}")
    print(f"CSV log: {csv_log}")


if __name__ == "__main__":
    main()
