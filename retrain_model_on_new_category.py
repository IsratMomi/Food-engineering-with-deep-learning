import os
import random
import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import transforms, models
from torchvision.datasets import Food101

import cv2
from PIL import Image
from tqdm import tqdm

# ---------------- CONFIG ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0

DEFAULT_DATA_ROOT = r"E:\er\project-food\data"
DEFAULT_BANGLA_ROOT = r"E:\er\project-food\data\bangla-food"
DEFAULT_SYNTH_ROOT = os.path.join(DEFAULT_DATA_ROOT, "synthetic_data")

OLD_MODEL_PATH = r"./checkpoints_cv_food_bangla/resnet50_food101_bangla_120.pth"
NEW_SAVE_PATH = r"./checkpoints_cv_food_bangla/resnet50_food101_bangla_UPDATED.pth"


# ---------------- DATASET ----------------
class SimpleImageDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self.samples))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label


# ---------------- HELPERS ----------------
def build_resnet(num_classes: int):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def collect_imagefolder_samples(root_dir: str, class_offset: int = 0):
    classes = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    cls_to_idx = {c: i + class_offset for i, c in enumerate(classes)}

    samples = []
    for cls in classes:
        cdir = os.path.join(root_dir, cls)
        for fname in os.listdir(cdir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.abspath(os.path.join(cdir, fname)), cls_to_idx[cls]))

    return samples, classes, cls_to_idx


def load_synthetic_samples(syn_root: str, class_to_idx: dict):
    if not os.path.isdir(syn_root):
        return []

    out = []
    for cls_name in os.listdir(syn_root):
        cls_dir = os.path.join(syn_root, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        if cls_name not in class_to_idx:
            continue

        label = class_to_idx[cls_name]

        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                out.append((os.path.abspath(os.path.join(cls_dir, fname)), label))

    return out


def freeze_backbone(model):
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), 100.0 * correct / total


def eval_model(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Testing", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100.0 * correct / total


# ---------------- MAIN ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    print("📦 Loading Food101...")
    food_train = Food101(root=DEFAULT_DATA_ROOT, split="train", download=False)
    food_test = Food101(root=DEFAULT_DATA_ROOT, split="test", download=False)

    food_classes = food_train.classes
    num_food = len(food_classes)

    food_train_samples = [(os.path.abspath(p), int(y)) for p, y in zip(food_train._image_files, food_train._labels)]
    food_test_samples = [(os.path.abspath(p), int(y)) for p, y in zip(food_test._image_files, food_test._labels)]

    food_class_to_idx = {c: i for i, c in enumerate(food_classes)}

    print("📦 Loading Bangla dataset...")
    bangla_train_root = os.path.join(DEFAULT_BANGLA_ROOT, "train")
    bangla_test_root = os.path.join(DEFAULT_BANGLA_ROOT, "test")

    bangla_train_samples, bangla_classes, _ = collect_imagefolder_samples(
        bangla_train_root, class_offset=num_food
    )
    bangla_test_samples, _, _ = collect_imagefolder_samples(
        bangla_test_root, class_offset=num_food
    )

    num_bangla = len(bangla_classes)
    total_classes = num_food + num_bangla

    print(f"✅ Food classes: {num_food}")
    print(f"✅ Bangla classes: {num_bangla}")
    print(f"✅ Total classes now: {total_classes}")

    print("📦 Loading synthetic images...")
    synth_samples = load_synthetic_samples(DEFAULT_SYNTH_ROOT, food_class_to_idx)
    print(f"✅ Synthetic images: {len(synth_samples)}")

    # Merge training dataset (IMPORTANT to avoid forgetting)
    merged_train_samples = food_train_samples + synth_samples + bangla_train_samples
    merged_train_labels = [y for _, y in food_train_samples] + [y for _, y in synth_samples] + [y for _, y in bangla_train_samples]

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.RandomResizedCrop(256, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_ds = SimpleImageDataset(merged_train_samples, transform=train_transform)
    food_test_ds = SimpleImageDataset(food_test_samples, transform=val_transform)
    bangla_test_ds = SimpleImageDataset(bangla_test_samples, transform=val_transform)

    # Weighted Sampler
    counts = np.bincount(merged_train_labels, minlength=total_classes)
    weights = 1.0 / (counts + 1e-6)
    sample_weights = [float(weights[y]) for y in merged_train_labels]

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=NUM_WORKERS)
    food_test_loader = DataLoader(food_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS)
    bangla_test_loader = DataLoader(bangla_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS)

    # Load old model (120-class) and expand FC
    print(f"🔁 Loading old trained model: {OLD_MODEL_PATH}")
    old_ckpt = torch.load(OLD_MODEL_PATH, map_location=DEVICE, weights_only=False)

    old_num_food = old_ckpt["num_food_classes"]
    old_num_bangla = old_ckpt["num_bangla_classes"]
    old_total = old_num_food + old_num_bangla

    print(f"Old model classes: {old_total}")
    print(f"New required classes: {total_classes}")

    old_model = build_resnet(old_total).to(DEVICE)
    old_model.load_state_dict(old_ckpt["model_state_dict"])

    new_model = build_resnet(total_classes).to(DEVICE)

    # Copy backbone weights
    new_sd = new_model.state_dict()
    old_sd = old_model.state_dict()

    for k in old_sd:
        if k.startswith("fc."):
            continue
        new_sd[k].copy_(old_sd[k])

    # Copy old FC weights into new FC
    new_sd["fc.weight"][:old_total].copy_(old_sd["fc.weight"])
    new_sd["fc.bias"][:old_total].copy_(old_sd["fc.bias"])

    new_model.load_state_dict(new_sd)

    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Stage 1: Freeze backbone (avoid forgetting)
    print("🧊 Stage 1: Freezing backbone (train only FC)")
    freeze_backbone(new_model)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, new_model.parameters()), lr=args.lr)

    freeze_epochs = 5
    for epoch in range(freeze_epochs):
        loss, acc = train_one_epoch(new_model, train_loader, optimizer, criterion)
        food_acc = eval_model(new_model, food_test_loader)
        bangla_acc = eval_model(new_model, bangla_test_loader)
        print(f"[Frozen] Epoch {epoch+1}/{freeze_epochs} Loss={loss:.4f} TrainAcc={acc:.2f}% "
              f"| FoodAcc={food_acc:.2f}% | BanglaAcc={bangla_acc:.2f}%")

    # Stage 2: Unfreeze all layers
    print("🔥 Stage 2: Unfreezing all layers (fine-tune full model)")
    unfreeze_all(new_model)

    optimizer = optim.Adam(new_model.parameters(), lr=args.lr / 2)

    for epoch in range(args.epochs):
        loss, acc = train_one_epoch(new_model, train_loader, optimizer, criterion)
        food_acc = eval_model(new_model, food_test_loader)
        bangla_acc = eval_model(new_model, bangla_test_loader)

        print(f"[Finetune] Epoch {epoch+1}/{args.epochs} Loss={loss:.4f} TrainAcc={acc:.2f}% "
              f"| FoodAcc={food_acc:.2f}% | BanglaAcc={bangla_acc:.2f}%")

    # Save updated model
    torch.save({
        "model_state_dict": new_model.state_dict(),
        "num_food_classes": num_food,
        "num_bangla_classes": num_bangla,
        "bangla_classes": bangla_classes
    }, NEW_SAVE_PATH)

    print(f"✅ Updated model saved to: {NEW_SAVE_PATH}")


if __name__ == "__main__":
    main()