# predict_single.py
# ✅ Consistent with training: checkpoint + val_transform

import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.datasets import Food101
from torchvision import transforms
from PIL import Image
import os

# ====== Configuration ======
IMAGE_PATH = "./data/sample-data/pasta2.png"   # your custom test image
MODEL_PATH = "./checkpoints/resnet50_food101_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Load class labels from Food101 ======
class_names = Food101(root="./data", download=False).classes

# ====== Define SAME val_transform as training ======
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ====== Load trained ResNet50 model ======
model = models.resnet50(weights=None)  # no pretrained weights
model.fc = torch.nn.Linear(model.fc.in_features, 101)

# ✅ load checkpoint exactly as training script saved
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

# ====== Load and process image ======
image = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = val_transform(image).unsqueeze(0).to(DEVICE)

# ====== Inference ======
with torch.no_grad():
    outputs = model(image_tensor)
    probs = F.softmax(outputs, dim=1)
    top3_probs, top3_idxs = torch.topk(probs, k=3, dim=1)

# ====== Show Top-3 Predictions ======
print("\n🍽️ Top-3 Predictions:")
for i in range(3):
    label = class_names[top3_idxs[0][i].item()]
    confidence = top3_probs[0][i].item() * 100
    print(f"  {i+1}. {label} — {confidence:.2f}%")

# ====== Warning if Low Confidence ======
if top3_probs[0][0].item() * 100 < 40:
    print("⚠️ Warning: Low prediction confidence. Might be an unknown class.")