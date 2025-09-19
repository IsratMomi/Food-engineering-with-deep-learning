# resnet50_train.py (Full image processing pipeline: pre- and post-training)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import Food101
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import os
from tqdm import tqdm

# ====== Config ======
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "./checkpoints/resnet50_food101.pth"

# ====== Preprocessing Function (Training-time) ======
def preprocess_cv2(img: np.ndarray):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced_img

# ====== Post-processing Function (Post-training enhancement) ======
def post_process_image(img_pil):
    img_pil = img_pil.filter(ImageFilter.SHARPEN)
    img_pil = ImageEnhance.Brightness(img_pil).enhance(1.2)
    img_np = np.array(img_pil)
    img_np = cv2.fastNlMeansDenoisingColored(img_np, None, 10, 10, 7, 21)
    return Image.fromarray(img_np)

# ====== Custom Dataset Wrapper ======
class Food101Preprocessed(Food101):
    def __getitem__(self, index):
        path, target = self.samples[index]
        img_cv = cv2.imread(path)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_cv = preprocess_cv2(img_cv)
        img_pil = Image.fromarray(img_cv)
        img_pil = post_process_image(img_pil)  # apply post-processing also during training
        if self.transform is not None:
            img_pil = self.transform(img_pil)
        return img_pil, target

# ====== Transforms ======
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ====== Load Datasets with Augmented Preprocessing ======
train_dataset = Food101Preprocessed(root="./data", split="train", download=True, transform=train_transform)
val_dataset = Food101Preprocessed(root="./data", split="test", download=False, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ====== Model ======
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 101)
model = model.to(DEVICE)

# ====== Loss and Optimizer ======
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ====== Resume Checkpoint ======
start_epoch = 0
if os.path.exists(SAVE_PATH):
    print("🔁 Resuming from checkpoint...")
    checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"✅ Resumed from epoch {start_epoch}")

# ====== Training Loop ======
for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    print(f"\n🔁 Epoch {epoch+1}/{NUM_EPOCHS}")
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    print(f"🟢 Train Loss: {total_loss:.4f} | Train Acc: {train_acc:.2f}%")

    # ====== Validation ======
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    print(f"🔵 Validation Accuracy: {val_acc:.2f}%")

    # ====== Save Checkpoint ======
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, SAVE_PATH)
    print(f"💾 Checkpoint saved at epoch {epoch+1} → {SAVE_PATH}")