# dalle_image_generator.py
# ✅ Auto-generate and download food images using OpenAI DALL·E API (with automatic class selection)

import openai  # OpenAI SDK
import os
import requests
import sys
from time import sleep
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()

# ====== Configuration ======
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("❌ OPENAI_API_KEY not found. Set it as an environment variable.")
    sys.exit(1)

image_data_root = "data/food-101/images"  # original image dataset
output_dir = "data/synthetic_data"         # where to save generated images
images_per_prompt = 5
image_size = "1024x1024"

# ====== Auto-detect low-sample classes ======
class_counts = {
    cls: len(os.listdir(os.path.join(image_data_root, cls)))
    for cls in os.listdir(image_data_root)
    if os.path.isdir(os.path.join(image_data_root, cls))
}

# Sort by class count (ascending)
low_classes = sorted(class_counts.items(), key=lambda x: x[1])[:10]  # top 10 lowest
prompt_map = {
    cls: f"A high-resolution, realistic photo of freshly served {cls.replace('_', ' ')}, plated beautifully"
    for cls, count in low_classes
}

# ====== Generate and Save Images ======
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

for class_name, prompt in prompt_map.items():
    class_folder = os.path.join(output_dir, class_name)
    ensure_dir(class_folder)

    print(f"🎨 Generating images for: {class_name}")
    for i in range(images_per_prompt):
        try:
            response = openai.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size=image_size
            )
            image_url = response.data[0].url
            image_data = requests.get(image_url).content
            with open(os.path.join(class_folder, f"{class_name}_{i+1}.jpg"), 'wb') as img_file:
                img_file.write(image_data)
            print(f"  ✅ Saved: {class_name}_{i+1}.jpg")
            sleep(1)  # avoid rate limiting
        except Exception as e:
            print(f"  ❌ Error for {class_name} [{i+1}]: {e}")

print("\n✅ All synthetic images saved to ./synthetic_data/")