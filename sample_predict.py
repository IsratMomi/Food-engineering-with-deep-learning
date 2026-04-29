import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.datasets import Food101
from torchvision import transforms
from PIL import Image
import os

# ====== Configuration ======
IMAGE_PATH = "./data/sample-data/chotputi.jpg"
MODEL_PATH = "./checkpoints_cv_food_bangla/resnet50_food101_bangla_120.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Load Food101 classes ======
food_classes = Food101(root="./data", download=False).classes

# ====== Load Bangla classes (from TRAIN folder!) ======
bangla_train_root = r"E:\er\project-food\data\bangla-food\train"
bangla_classes = sorted(os.listdir(bangla_train_root))

# Combine in SAME order as training
class_names = food_classes + bangla_classes
num_classes = len(class_names)

print(f"Total classes loaded: {num_classes}")

# ====== SAME val_transform as training ======
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ====== Build 120-class model ======
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# ====== Load checkpoint ======
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")

# ====== Load and process image ======
image = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = val_transform(image).unsqueeze(0).to(DEVICE)

# ====== Inference ======
with torch.no_grad():
    outputs = model(image_tensor)
    probs = F.softmax(outputs, dim=1)
    top5_probs, top5_idxs = torch.topk(probs, k=5, dim=1)

# ====== Show Top-5 Predictions ======
print("\n🍽️ Top-5 Predictions:")
for i in range(5):
    label = class_names[top5_idxs[0][i].item()]
    confidence = top5_probs[0][i].item() * 100
    print(f"  {i+1}. {label} — {confidence:.2f}%")

if top5_probs[0][0].item() * 100 < 40:
    print("⚠️ Low confidence prediction.")