import torch
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# ----------------------------
# 1. Load Food-101 class names
# ----------------------------
# Torchvision's Food101 dataset has classes in fixed order
food101 = datasets.Food101(root="./data", split="train", download=True)
idx_to_class = {i: c for i, c in enumerate(food101.classes)}

# ----------------------------
# 2. Load your trained model
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet50(pretrained=False)
num_classes = 101
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load your trained checkpoint
ckpt_path = "./checkpoints/resnet50_food101_best.pth"
state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(state["model_state_dict"])  # because your script saves dict with keys
model.to(DEVICE)
model.eval()

# ----------------------------
# 3. Preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# img_path = "steak.png"   # your image
img_path = "./data/sample-data/steak1.jpg" 
img = Image.open(img_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(DEVICE)

# ----------------------------
# 4. Hooks for Grad-CAM
# ----------------------------
gradients = []
activations = []

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0].detach())

def forward_hook(module, input, output):
    activations.append(output.detach())

target_layer = model.layer4[2].conv3
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# ----------------------------
# 5. Forward + backward
# ----------------------------
output = model(input_tensor)
pred_class = output.argmax(dim=1).item()
pred_label = idx_to_class[pred_class]

model.zero_grad()
loss = output[0, pred_class]
loss.backward()

# ----------------------------
# 6. Grad-CAM heatmap
# ----------------------------
grads = gradients[0].mean(dim=(2, 3), keepdim=True)  # global average pooling
activation = activations[0]

cam = (activation * grads).sum(dim=1, keepdim=True)
cam = F.relu(cam)
cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
cam = cam.squeeze().cpu().numpy()

cam = (cam - cam.min()) / (cam.max() - cam.min())  # normalize [0,1]

# ----------------------------
# 7. Overlay visualization
# ----------------------------
img_cv = cv2.cvtColor(np.array(img.resize((224, 224))), cv2.COLOR_RGB2BGR)
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = np.float32(heatmap) / 255 + np.float32(img_cv) / 255
overlay = overlay / overlay.max()

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Grad-CAM Heatmap")
plt.imshow(cam, cmap="jet")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title(f"Overlay → {pred_label}")
plt.imshow(overlay[..., ::-1])  # BGR→RGB
plt.axis("off")

plt.show()

print(f"Predicted class index: {pred_class} → {pred_label}")