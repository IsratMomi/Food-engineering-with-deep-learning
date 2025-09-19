# dalle_image_generator.py
# ✅ Auto-generate and download food images using OpenAI DALL·E API (with automatic class selection)
# ✅ Budget-aware: stays within a user-defined USD budget and touches every selected class at least once

import openai  # OpenAI SDK
import os
import requests
import sys
from time import sleep
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()

# ====== Configuration ======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY not found. Set it as an environment variable.")
    sys.exit(1)
openai.api_key = OPENAI_API_KEY

# Paths
image_data_root = "data/food-101/images"  # original image dataset
output_dir = "data/synthetic_data"         # where to save generated images

# DALL·E settings
image_size = "1024x1024"  # valid: '1024x1024', '1024x1792', '1792x1024'
model_name = "dall-e-3"

# Budget settings (USD)
BUDGET_USD = 5.00  # total spend cap
# Approximate cost per image by size (update if OpenAI pricing changes)
COST_PER_IMAGE = {
    "1024x1024": 0.08,
    "1024x1792": 0.12,
    "1792x1024": 0.12,
}

# Safety pause to avoid rate limits (seconds)
SLEEP_SEC = 1.0

# ====== Helpers ======
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_low_sample_classes(root, limit=10):
    classes = []
    for cls in os.listdir(root):
        p = os.path.join(root, cls)
        if os.path.isdir(p):
            try:
                count = len([f for f in os.listdir(p) if f.lower().endswith((".jpg",".jpeg",".png"))])
            except Exception:
                count = 0
            classes.append((cls, count))
    classes.sort(key=lambda x: x[1])
    return classes[:limit]

# ====== Select classes (lowest sample counts) ======
LOW_CLASS_LIMIT = 10  # you can change this; script will still stay within budget
low_classes = get_low_sample_classes(image_data_root, limit=LOW_CLASS_LIMIT)
if not low_classes:
    print(f"❌ No class folders found under {image_data_root}")
    sys.exit(1)

# Build prompt map for each class
prompt_map = {
    cls: f"A high-resolution, realistic food photography shot of freshly served {cls.replace('_',' ')}, plated beautifully on a table"
    for cls, _ in low_classes
}

# ====== Budget planning ======
cost_per_img = COST_PER_IMAGE.get(image_size)
if cost_per_img is None:
    print(f"❌ Unsupported image_size: {image_size}")
    sys.exit(1)

num_classes = len(prompt_map)
max_images_total = int(BUDGET_USD // cost_per_img)  # max images we can afford in total
if max_images_total < num_classes:
    print("⚠️ Budget too small to generate at least 1 image per class.")
    print(f"   Budget=${BUDGET_USD:.2f}, cost/image=${cost_per_img:.2f}, classes={num_classes} → max_images_total={max_images_total}")
    print("   Increase budget or reduce number of classes.")
    sys.exit(1)

# Distribute images across classes so each gets at least 1
base_per_class = max_images_total // num_classes
remainder = max_images_total % num_classes

# At least one per class
per_class_counts = {cls: max(1, base_per_class) for cls in prompt_map.keys()}
# Distribute the remainder one-by-one
for cls in list(prompt_map.keys())[:remainder]:
    per_class_counts[cls] += 1

estimated_cost = sum(per_class_counts.values()) * cost_per_img
print("💰 Budget plan:")
print(f"   Target classes: {num_classes}")
print(f"   Total images planned: {sum(per_class_counts.values())}")
print(f"   Estimated cost (@${cost_per_img:.2f}/img): ${estimated_cost:.2f} (<= ${BUDGET_USD:.2f})")

# ====== Generate and Save Images (budget-aware) ======
ensure_dir(output_dir)

spent_images = 0
for class_name, prompt in prompt_map.items():
    n_images = per_class_counts[class_name]
    class_folder = os.path.join(output_dir, class_name)
    ensure_dir(class_folder)
    print(f"🎨 Generating {n_images} image(s) for: {class_name}")

    for i in range(n_images):
        # Additional runtime guard: stop if next image would exceed budget
        if (spent_images + 1) * cost_per_img > BUDGET_USD + 1e-9:
            print("⛔ Budget cap reached. Stopping generation.")
            sys.exit(0)
        try:
            response = openai.images.generate(
                model=model_name,
                prompt=prompt,
                n=1,
                size=image_size
            )
            image_url = response.data[0].url
            image_data = requests.get(image_url, timeout=30).content
            outfile = os.path.join(class_folder, f"{class_name}_{i+1}.jpg")
            with open(outfile, 'wb') as img_file:
                img_file.write(image_data)
            spent_images += 1
            print(f"  ✅ Saved: {outfile}")
            sleep(SLEEP_SEC)
        except Exception as e:
            # Handle common billing limit error gracefully
            msg = str(e)
            if "billing_hard_limit_reached" in msg:
                print("⛔ OpenAI billing hard limit reached. Stopping early.")
                sys.exit(0)
            print(f"  ❌ Error for {class_name} [{i+1}]: {e}")

print("✅ All synthetic images saved to ./synthetic_data/ within budget.")