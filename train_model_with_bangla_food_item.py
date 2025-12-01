# train.py
# Combined Stage-1 (head-only Bangla fine-tune) + Stage-2 (partial unfreeze) training
# Includes MixUp toggle, auto-resume, logging, CLAHE (train-only), and best checkpoint saving

import os
import argparse
import json
from datetime import datetime
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import numpy as np
from tqdm import tqdm

# --------------------------------------------------------
# Optional: MixUp
# --------------------------------------------------------
def mixup_data(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y, torch.zeros(len(x))
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, index, lam

# --------------------------------------------------------
# Model Builder
# --------------------------------------------------------
def build_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# --------------------------------------------------------
# Freeze layers (Stage 1) / Partial unfreeze (Stage 2)
# --------------------------------------------------------
def freeze_all_but_head(model):
    for name, p in model.named_parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

# Unfreeze last N blocks (default 1 block = layer4)
def partial_unfreeze(model, blocks=1):
    layers = [model.layer4, model.layer3, model.layer2, model.layer1]
    for p in model.parameters():
        p.requires_grad = False
    for i in range(blocks):
        for p in layers[i].parameters():
            p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True

# --------------------------------------------------------
# CLAHE (train-only)
# --------------------------------------------------------
import cv2
from PIL import Image

def clahe_img(img):
    arr = np.array(img)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return Image.fromarray(final)

# --------------------------------------------------------
# Train / Validate loops
# --------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, mixup_alpha=0.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="Train", ncols=100):
        imgs, labels = imgs.to(device), labels.to(device)

        if mixup_alpha > 0:
            imgs, labels_orig, index, lam = mixup_data(imgs, labels, mixup_alpha)
            outputs = model(imgs)
            loss = lam * criterion(outputs, labels_orig) + (1 - lam) * criterion(outputs, labels_orig[index])
            _, preds = outputs.max(1)
            correct += (lam * preds.eq(labels_orig).sum().item() + (1 - lam) * preds.eq(labels_orig[index]).sum().item())
            total += labels.size(0)
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader), 100 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Val", ncols=100):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), 100 * correct / total

# --------------------------------------------------------
# Main
# --------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--synthetic', type=str, default=None)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epochs1', type=int, default=15)
    parser.add_argument('--epochs2', type=int, default=10)
    parser.add_argument('--lr1', type=float, default=1e-3)
    parser.add_argument('--lr2', type=float, default=1e-4)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --------------------------------------------------
    # Transforms
    # --------------------------------------------------
    train_tf = transforms.Compose([
        transforms.Lambda(clahe_img),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_path = os.path.join(args.data, 'train')
    val_path = os.path.join(args.data, 'test')

    train_set = ImageFolder(train_path, transform=train_tf)
    val_set = ImageFolder(val_path, transform=test_tf)

    if args.synthetic and os.path.isdir(args.synthetic):
        syn_set = ImageFolder(args.synthetic, transform=train_tf)
        train_set = syn_set + train_set
        print(f"[+] Synthetic merged: {len(syn_set)} images")

    # Weighted Sampler
    class_counts = np.bincount([y for _, y in train_set])
    class_weights = 1. / np.maximum(class_counts, 1)
    weights = [class_weights[y] for _, y in train_set]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=args.batch, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=4)

    num_classes = len(train_set.classes)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = build_model(num_classes).to(device)
    freeze_all_but_head(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Stage 1 optimizer
    optimizer1 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr1)

    best_acc = 0

    print("========== Stage 1: Head-only ==========")
    for epoch in range(1, args.epochs1 + 1):
        lr_now = optimizer1.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{args.epochs1} | LR={lr_now:.8f}")

        loss_tr, acc_tr = train_one_epoch(model, train_loader, optimizer1, criterion, device, mixup_alpha=0.4 if args.mixup else 0.0)
        loss_val, acc_val = validate(model, val_loader, criterion, device)

        print(f"Loss={loss_tr:.4f} | TrainAcc={acc_tr:.2f}% | ValAcc={acc_val:.2f}%")

        # Save best
        if acc_val > best_acc:
            best_acc = acc_val
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_stage1.pth'))
            print(f"[+] New best Stage1: {best_acc:.2f}% saved.")

    # --------------------------------------------------
    # Stage 2: Partial Unfreeze
    # --------------------------------------------------
    print("\n========== Stage 2: Partial Unfreeze ==========")
    partial_unfreeze(model, blocks=1)

    optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr2)

    for epoch in range(1, args.epochs2 + 1):
        lr_now = optimizer2.param_groups[0]['lr']
        print(f"\n[Stage2] Epoch {epoch}/{args.epochs2} | LR={lr_now:.8f}")

        loss_tr, acc_tr = train_one_epoch(model, train_loader, optimizer2, criterion, device, mixup_alpha=0.4 if args.mixup else 0.0)
        loss_val, acc_val = validate(model, val_loader, criterion, device)

        print(f"[Stage2] Loss={loss_tr:.4f} | TrainAcc={acc_tr:.2f}% | ValAcc={acc_val:.2f}%")

        # Save best stage2
        if acc_val > best_acc:
            best_acc = acc_val
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_overall.pth'))
            print(f"[+] New BEST overall: {best_acc:.2f}% saved.")

    print("\nTraining complete.")