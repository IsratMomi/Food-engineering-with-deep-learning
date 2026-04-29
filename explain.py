"""
explain.py — Explainable AI for Food Classification (Food101 + Bangla)
=======================================================================
120 Classes: 101 Food101 + 19 Bangladeshi food classes

Features:
  1. Top-5 predictions with confidence scores
  2. Grad-CAM heatmap for the predicted (top) class
  3. Grad-CAM heatmaps for lower-ranked classes (why they scored lower)
  4. Feature similarity — nearest food classes based on FC weight cosine similarity

Usage:
  python explain.py --image path/to/food.jpg --model path/to/model.pth
  python explain.py --image path/to/food.jpg --model path/to/model.pth --arch resnet50
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────
# CLASS NAMES — Food101 (101) + Bangladeshi Foods (19) = 120 total
# ⚠️  Order must match your training label encoding exactly
# ─────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    # ── Food101 (101 classes, indices 0–100) — exact alphabetical folder order ──
    "Apple Pie",                # 0
    "Baby Back Ribs",           # 1
    "Baklava",                  # 2
    "Beef Carpaccio",           # 3
    "Beef Tartare",             # 4
    "Beet Salad",               # 5
    "Beignets",                 # 6
    "Bibimbap",                 # 7
    "Bread Pudding",            # 8
    "Breakfast Burrito",        # 9
    "Bruschetta",               # 10
    "Caesar Salad",             # 11
    "Cannoli",                  # 12
    "Caprese Salad",            # 13
    "Carrot Cake",              # 14
    "Ceviche",                  # 15
    "Cheese Plate",             # 16
    "Cheesecake",               # 17
    "Chicken Curry",            # 18
    "Chicken Quesadilla",       # 19
    "Chicken Wings",            # 20
    "Chocolate Cake",           # 21
    "Chocolate Mousse",         # 22
    "Churros",                  # 23
    "Clam Chowder",             # 24
    "Club Sandwich",            # 25
    "Crab Cakes",               # 26
    "Creme Brulee",             # 27
    "Croque Madame",            # 28
    "Cup Cakes",                # 29
    "Deviled Eggs",             # 30
    "Donuts",                   # 31
    "Dumplings",                # 32
    "Edamame",                  # 33
    "Eggs Benedict",            # 34
    "Escargots",                # 35
    "Falafel",                  # 36
    "Filet Mignon",             # 37
    "Fish And Chips",           # 38
    "Foie Gras",                # 39
    "French Fries",             # 40
    "French Onion Soup",        # 41
    "French Toast",             # 42
    "Fried Calamari",           # 43
    "Fried Rice",               # 44
    "Frozen Yogurt",            # 45
    "Garlic Bread",             # 46
    "Gnocchi",                  # 47
    "Greek Salad",              # 48
    "Grilled Cheese Sandwich",  # 49
    "Grilled Salmon",           # 50
    "Guacamole",                # 51
    "Gyoza",                    # 52
    "Hamburger",                # 53
    "Hot And Sour Soup",        # 54
    "Hot Dog",                  # 55
    "Huevos Rancheros",         # 56
    "Hummus",                   # 57
    "Ice Cream",                # 58
    "Lasagna",                  # 59
    "Lobster Bisque",           # 60
    "Lobster Roll Sandwich",    # 61
    "Macaroni And Cheese",      # 62
    "Macarons",                 # 63
    "Miso Soup",                # 64
    "Mussels",                  # 65
    "Nachos",                   # 66
    "Omelette",                 # 67
    "Onion Rings",              # 68
    "Oysters",                  # 69
    "Pad Thai",                 # 70
    "Paella",                   # 71
    "Pancakes",                 # 72
    "Panna Cotta",              # 73
    "Peking Duck",              # 74
    "Pho",                      # 75
    "Pizza",                    # 76
    "Pork Chop",                # 77
    "Poutine",                  # 78
    "Prime Rib",                # 79
    "Pulled Pork Sandwich",     # 80
    "Ramen",                    # 81
    "Ravioli",                  # 82
    "Red Velvet Cake",          # 83
    "Risotto",                  # 84
    "Samosa",                   # 85
    "Sashimi",                  # 86
    "Scallops",                 # 87
    "Seaweed Salad",            # 88
    "Shrimp And Grits",         # 89
    "Spaghetti Bolognese",      # 90
    "Spaghetti Carbonara",      # 91
    "Spring Rolls",             # 92
    "Steak",                    # 93
    "Strawberry Shortcake",     # 94
    "Sushi",                    # 95
    "Tacos",                    # 96
    "Takoyaki",                 # 97
    "Tiramisu",                 # 98
    "Tuna Tartare",             # 99
    "Waffles",                  # 100

    # ── Bangladeshi Foods (20 classes, indices 101–120) — exact alphabetical folder order ──
    "Bakorkhani",               # 101
    "Beguni",                   # 102
    "Biriyani",                 # 103
    "Chickpeas",                # 104
    "Chotpoti",                 # 105
    "Egg Omlette",              # 106
    "Fuchka",                   # 107
    "Haleem",                   # 108
    "Hilsha Fish",              # 109
    "Kabab",                    # 110
    "Kacha Golla",              # 111
    "Kala Bhuna",               # 112
    "Khichuri",                 # 113
    "Mashed Potato",            # 114
    "Morog Polao",              # 115
    "Nehari",                   # 116
    "Porota",                   # 117
    "Roshgolla",                # 118
    "Roshmalai",                # 119
    "Yogurt",                   # 120
]

assert len(CLASS_NAMES) == 121, f"Expected 121 classes, got {len(CLASS_NAMES)}"

OUTPUT_DIR = Path("xai_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Bangladeshi class indices (for color coding in charts)
BANGLA_INDICES = set(range(101, 121))


# ─────────────────────────────────────────────
# IMAGE TRANSFORM
# ─────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = output[0][class_idx]
        target.backward()

        pooled_grads = self.gradients.mean(dim=[0, 2, 3])
        activation = self.activations[0].clone()

        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_grads[i]

        heatmap = activation.mean(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() != 0:
            heatmap /= heatmap.max()
        return heatmap


def overlay_heatmap(heatmap, original_img, alpha=0.5, colormap=plt.cm.jet):
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(255 * heatmap)).resize((224, 224), Image.LANCZOS)
    ) / 255.0
    img_np = np.array(original_img.resize((224, 224))) / 255.0
    colored = colormap(heatmap_resized)[:, :, :3]
    overlaid = alpha * colored + (1 - alpha) * img_np
    return np.clip(overlaid, 0, 1)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
def load_model(model_path, arch, num_classes, device):
    arch_fn = getattr(models, arch)
    model = arch_fn(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(model_path, map_location=device)
    # Support both raw state dict and checkpoint dict
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def get_target_layer(model, arch):
    if "resnet" in arch:
        return model.layer4[-1]
    raise ValueError(f"Unsupported arch '{arch}'. Set target_layer manually.")


# ─────────────────────────────────────────────
# PREDICT TOP-5
# ─────────────────────────────────────────────
def predict_top5(model, input_tensor):
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]
    top5_probs, top5_indices = torch.topk(probs, 5)
    results = [
        (CLASS_NAMES[idx.item()], top5_probs[i].item(), idx.item())
        for i, idx in enumerate(top5_indices)
    ]
    return results, probs.cpu().numpy()


# ─────────────────────────────────────────────
# FEATURE SIMILARITY
# ─────────────────────────────────────────────
def compute_class_similarity(model, probs):
    """
    Cosine similarity between the top predicted class FC weight vector
    and all other class weight vectors.
    Shows which food classes share the most similar learned features.
    """
    fc_weights = model.fc.weight.detach().cpu().numpy()  # (120, feat_dim)
    top_class_idx = int(np.argmax(probs))
    top_vec = fc_weights[top_class_idx].reshape(1, -1)
    similarities = cosine_similarity(top_vec, fc_weights)[0]

    sim_ranked = sorted(
        [
            (CLASS_NAMES[i], float(similarities[i]), float(probs[i]), i)
            for i in range(len(CLASS_NAMES))
            if i != top_class_idx
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    return sim_ranked


# ─────────────────────────────────────────────
# CONSOLE SUMMARY
# ─────────────────────────────────────────────
def print_summary(top5, sim_ranked):
    top_name, top_conf, top_idx = top5[0]

    print("\n" + "=" * 65)
    print("  TOP-5 PREDICTIONS")
    print("=" * 65)
    for rank, (name, conf, idx) in enumerate(top5):
        tag    = "✅" if rank == 0 else "  "
        origin = "[Bangla] " if idx in BANGLA_INDICES else "[Food101]"
        bar    = "█" * int(conf * 40)
        print(f"  {tag} #{rank+1}  {name:<30} {origin}  {conf*100:5.1f}%  {bar}")

    print("\n" + "=" * 65)
    print(f"  TOP-10 NEAREST CLASSES TO '{top_name}' (feature similarity)")
    print("=" * 65)
    for name, sim, prob, idx in sim_ranked[:10]:
        origin = "[Bangla] " if idx in BANGLA_INDICES else "[Food101]"
        bar    = "█" * int(sim * 30)
        print(f"  {name:<30} {origin}  sim={sim:.3f}  conf={prob*100:4.1f}%  {bar}")

    print("\n" + "=" * 65)
    print("  WHY LOWER CLASSES SCORED LESS")
    print("=" * 65)
    for rank, (name, conf, idx) in enumerate(top5[1:], 2):
        print(f"  #{rank}  {name:<30} {conf*100:5.1f}%  "
              f"← fewer '{top_name}' visual features in focus region")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────
# FULL VISUAL XAI REPORT
# ─────────────────────────────────────────────
def visualize(image_path, original_img, top5, sim_ranked, gradcam, input_tensor):
    stem      = Path(image_path).stem
    num_lower = min(4, len(top5) - 1)

    fig = plt.figure(figsize=(24, 16))
    fig.patch.set_facecolor("#0f0f0f")

    gs = gridspec.GridSpec(
        3, 5, figure=fig,
        hspace=0.5, wspace=0.35,
        left=0.04, right=0.98,
        top=0.93, bottom=0.05,
    )

    # ── Row 0 | Col 0 : Original image ──
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(np.array(original_img.resize((224, 224))))
    ax_orig.set_title("Input Image", color="white", fontsize=10, pad=4)
    ax_orig.axis("off")

    # ── Row 0 | Col 1 : Grad-CAM — top predicted class ──
    top_name, top_conf, top_idx = top5[0]
    heatmap_top  = gradcam.generate(input_tensor, top_idx)
    overlay_top  = overlay_heatmap(heatmap_top, original_img, colormap=plt.cm.jet)
    origin_tag   = "[Bangla]" if top_idx in BANGLA_INDICES else "[Food101]"

    ax_top = fig.add_subplot(gs[0, 1])
    ax_top.imshow(overlay_top)
    ax_top.set_title(
        f"✅ {top_name} {origin_tag}\n{top_conf*100:.1f}% confidence",
        color="#00ff88", fontsize=9, pad=4,
    )
    ax_top.axis("off")

    # ── Row 0 | Cols 2–4 : Top-5 confidence bar chart ──
    ax_bar = fig.add_subplot(gs[0, 2:])
    names  = [r[0] for r in top5]
    confs  = [r[1] * 100 for r in top5]
    colors = ["#00ff88"] + ["#ff6b6b"] * num_lower

    bars = ax_bar.barh(names[::-1], confs[::-1],
                       color=colors[::-1], edgecolor="none", height=0.55)
    for bar, conf in zip(bars, confs[::-1]):
        ax_bar.text(bar.get_width() + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{conf:.1f}%", va="center", color="white", fontsize=9)

    ax_bar.set_xlim(0, 110)
    ax_bar.set_xlabel("Confidence (%)", color="white", fontsize=9)
    ax_bar.set_title("Top-5 Predictions", color="white", fontsize=11)
    ax_bar.tick_params(colors="white", labelsize=8)
    ax_bar.set_facecolor("#1a1a1a")
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#333")

    # ── Row 1 | Cols 0–3 : Grad-CAM for lower-ranked classes ──
    for j in range(num_lower):
        lower_name, lower_conf, lower_idx = top5[j + 1]
        heatmap_lower = gradcam.generate(input_tensor, lower_idx)
        overlay_lower = overlay_heatmap(heatmap_lower, original_img, colormap=plt.cm.plasma)
        lower_tag     = "[Bangla]" if lower_idx in BANGLA_INDICES else "[Food101]"

        ax_lower = fig.add_subplot(gs[1, j])
        ax_lower.imshow(overlay_lower)
        ax_lower.set_title(
            f"❌ {lower_name} {lower_tag}\n{lower_conf*100:.1f}% — wrong region focus",
            color="#ff6b6b", fontsize=8, pad=4,
        )
        ax_lower.axis("off")

    # ── Row 1 | Col 4 : Explanation text panel ──
    ax_txt = fig.add_subplot(gs[1, 4])
    ax_txt.set_facecolor("#1a1a1a")
    ax_txt.axis("off")
    explanation = (
        "WHY LOWER CLASSES\nSCORED LESS\n\n"
        "Each heatmap shows where\n"
        "the model looked when\n"
        "considering that class.\n\n"
        "🔴 Bright = strong attention\n"
        "🔵 Dark   = ignored region\n\n"
        "Lower classes activate on\n"
        "regions that are less\n"
        "relevant to the true food.\n\n"
        "Top class focuses sharply\n"
        "on the most distinctive\n"
        "visual features."
    )
    ax_txt.text(
        0.05, 0.97, explanation,
        transform=ax_txt.transAxes,
        fontsize=8.5, color="white", va="top", linespacing=1.5,
        bbox=dict(facecolor="#2a2a2a", edgecolor="#555", boxstyle="round,pad=0.6"),
    )

    # ── Row 2 | Full width : Feature similarity chart ──
    ax_sim   = fig.add_subplot(gs[2, :])
    top10    = sim_ranked[:10]
    s_names  = [r[0] for r in top10]
    s_sims   = [r[1] for r in top10]
    s_probs  = [r[2] * 100 for r in top10]
    s_idx    = [r[3] for r in top10]

    c_sim  = ["#f0a500" if i in BANGLA_INDICES else "#4ecdc4" for i in s_idx]
    c_conf = ["#f07800" if i in BANGLA_INDICES else "#ff6b6b" for i in s_idx]

    x     = np.arange(len(s_names))
    width = 0.38

    ax_sim.bar(x - width / 2, s_sims, width,
               label="Feature Similarity to Predicted Class",
               color=c_sim, alpha=0.9, edgecolor="none")
    ax_sim.bar(x + width / 2, [p / 100 for p in s_probs], width,
               label="Model Confidence",
               color=c_conf, alpha=0.9, edgecolor="none")

    # Annotate bars
    for xi, (sim, prob) in enumerate(zip(s_sims, s_probs)):
        ax_sim.text(xi - width / 2, sim + 0.02, f"{sim:.2f}",
                    ha="center", va="bottom", color="white", fontsize=6.5)
        ax_sim.text(xi + width / 2, prob / 100 + 0.02, f"{prob:.1f}%",
                    ha="center", va="bottom", color="white", fontsize=6.5)

    ax_sim.set_xticks(x)
    ax_sim.set_xticklabels(s_names, rotation=35, ha="right",
                            color="white", fontsize=8)
    ax_sim.set_ylabel("Score (0 – 1)", color="white", fontsize=9)
    ax_sim.set_ylim(0, 1.15)
    ax_sim.set_title(
        f"Top-10 Nearest Classes to '{top_name}' in Feature Space"
        f"   |   🟡 Bangladeshi   🩵 Food101",
        color="white", fontsize=11,
    )
    ax_sim.tick_params(colors="white", labelsize=8)
    ax_sim.set_facecolor("#1a1a1a")
    ax_sim.legend(facecolor="#2a2a2a", labelcolor="white", fontsize=8, loc="upper right")
    for spine in ax_sim.spines.values():
        spine.set_edgecolor("#333")

    # ── Super title ──
    origin_label = "Bangladeshi Food" if top_idx in BANGLA_INDICES else "Food101"
    fig.suptitle(
        f"XAI Report  ·  Predicted: {top_name.upper()}  ({top_conf*100:.1f}%)  [{origin_label}]"
        f"  ·  120 Classes (Food101 + Bangladeshi)",
        color="white", fontsize=13, fontweight="bold",
    )

    save_path = OUTPUT_DIR / f"xai_{stem}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"📊 XAI report saved → {save_path}")
    return save_path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="XAI — Food101 + Bangladeshi Food Classification (120 classes)"
    )
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input food image")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained .pth model file")
    parser.add_argument("--arch", type=str, default="resnet50",
                        choices=["resnet18", "resnet34", "resnet50"],
                        help="Architecture used during training (default: resnet50)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Device   : {device}")
    print(f"📁 Image    : {args.image}")
    print(f"🧠 Model    : {args.model}")
    print(f"🏗  Arch     : {args.arch}")
    print(f"🍽  Classes  : {len(CLASS_NAMES)}  (101 Food101 + 20 Bangladeshi)\n")

    # Load model
    model = load_model(args.model, args.arch, len(CLASS_NAMES), device)

    # Grad-CAM setup
    target_layer = get_target_layer(model, args.arch)
    gradcam      = GradCAM(model, target_layer)

    # Load & preprocess image
    original_img  = Image.open(args.image).convert("RGB")
    input_tensor  = TRANSFORM(original_img).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    # 1. Top-5 predictions
    top5, probs = predict_top5(model, input_tensor)

    # 2. Feature similarity — top-10 nearest classes
    sim_ranked = compute_class_similarity(model, probs)

    # 3. Console summary
    print_summary(top5, sim_ranked)

    # 4. Full visual XAI report (saved to xai_outputs/)
    visualize(args.image, original_img, top5, sim_ranked, gradcam, input_tensor)


if __name__ == "__main__":
    main()