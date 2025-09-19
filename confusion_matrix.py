import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.datasets import Food101
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./checkpoints/resnet50_food101_best.pth"

def main():
    # --- Load labels ---
    class_names = Food101(root="./data", download=False).classes

    # --- Transforms ---
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # --- Dataset ---
    val_ds = Food101(root="./data", split="test", download=False, transform=val_transform)

    # ⚡ Quick mode: use only first N samples (uncomment if needed)
    # N = 2000
    # val_ds = torch.utils.data.Subset(val_ds, range(N))

    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # --- Model ---
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 101)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    # --- Collect predictions with progress ---
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating", unit="batch"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(20, 20))
    disp.plot(ax=ax, cmap="viridis", xticks_rotation=90, colorbar=False)
    plt.title("Food-101 Confusion Matrix")
    plt.tight_layout()

    # Save instead of just show (safe for long runs)
    plt.savefig("confusion_matrix.png", dpi=200)
    print("✅ Confusion matrix saved as confusion_matrix.png")

if __name__ == "__main__":
    main()