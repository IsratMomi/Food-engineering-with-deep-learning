import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

# ------------------- Prompt -------------------
def get_prompt():
    return (
        "RAW photo of Bangladeshi chotpoti on ceramic plate,"
        "chickpeas curry with diced potatoes,"
        "(chopped onion:1.5), (shredded boiled egg:1.5), (coriander leaves:1.2),"
        "tomato pieces, tamarind sauce, green chili,"
        "top-down overhead view, realistic food photography, sharp focus"
    )


NEGATIVE_PROMPT = (
    "whole egg, full boiled egg, egg half, egg slices, "
    "rice, biryani, pulao, fried rice, noodles, bread, pizza, burger, "
    "blurry, low quality, text, watermark, logo, deformed, cartoon, anime"
)

# ------------------- Helpers -------------------
def load_real_images(folder: Path, extensions=("*.png", "*.jpg", "*.jpeg")):
    """Load only manually collected images (anything NOT named chotpoti_*.png)."""
    real = []
    for ext in extensions:
        for p in folder.glob(ext):
            if not p.stem.startswith("chotpoti_"):
                real.append(p)
    return sorted(real)


def preprocess_image(image_path: Path, width: int, height: int) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((width, height), Image.LANCZOS)
    return img


# ------------------- Main -------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate chotpoti images guided by real references (img2img)",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("--out-dir", type=str,
        default=r"E:\er\project-food\data\bangla-food\test\chotpoti",
        help=(
            "Folder where generated images will be saved.\n"
            "For test: ...\\test\\chotpoti\n"
            "For test:  ...\\test\\chotpoti"
        )
    )
    parser.add_argument("--ref-dir", type=str, default=None,
        help=(
            "Folder containing real reference images for img2img.\n"
            "If not set, uses --out-dir (test mode).\n"
            "Set to your test folder when generating test images:\n"
            "  --ref-dir ...\\test\\chotpoti"
        )
    )
    parser.add_argument("--target-count", type=int, default=80,
        help="Total number of generated images wanted in --out-dir. Default: 80"
    )
    parser.add_argument("--model", type=str,
        default="runwayml/stable-diffusion-v1-5"
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=8.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42,
        help=(
            "Random seed. Use different seeds for train vs test\n"
            "to ensure generated images are distinct.\n"
            "  Train: --seed 42 (default)\n"
            "  Test:  --seed 999"
        )
    )
    parser.add_argument("--strength", type=float, default=0.6,
        help=(
            "img2img strength — how much to deviate from the reference image.\n"
            "  0.3 = stays very close to real image (low diversity)\n"
            "  0.6 = balanced variation (recommended)\n"
            "  0.85 = highly creative, loose reference\n"
            "Default: 0.6"
        )
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve reference image folder
    ref_dir = Path(args.ref_dir) if args.ref_dir else out_dir
    is_cross_dir = ref_dir != out_dir

    # Count only previously generated images in out_dir
    generated_images = list(out_dir.glob("chotpoti_*.png"))
    current_count = len(generated_images)
    remaining = max(0, args.target_count - current_count)

    print(f"\n{'='*50}")
    print(f"📂 Output dir   : {out_dir}")
    print(f"🖼️  Reference dir : {ref_dir}{' (test folder)' if is_cross_dir else ''}")
    print(f"📊 Already generated : {current_count}")
    print(f"🧪 Need to generate  : {remaining}")
    print(f"🎲 Seed         : {args.seed}")
    print(f"💪 Strength     : {args.strength}")
    print(f"{'='*50}\n")

    if remaining == 0:
        print("✅ Already reached target count. Nothing to generate.")
        return

    # Load real reference images from ref_dir
    real_images = load_real_images(ref_dir)
    if not real_images:
        print(f"❌ No real reference images found in: {ref_dir}")
        print("   Make sure manually collected images do NOT start with 'chotpoti_'.")
        print("   Example valid names: img001.jpg, real_1.png, photo_01.jpg etc.")
        return

    print(f"✅ Found {len(real_images)} real reference image(s):")
    for p in real_images:
        print(f"   - {p.name}")
    print()

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"🚀 Loading model on {device}...")

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    prompt = get_prompt()
    rng = random.Random(args.seed)

    # Generate images
    for i in tqdm(range(remaining), desc="Generating chotpoti (img2img)"):
        # Cycle through real reference images
        ref_path = real_images[i % len(real_images)]
        ref_img = preprocess_image(ref_path, args.width, args.height)

        seed = args.seed + i
        g = torch.Generator(device=device).manual_seed(seed)

        # Slightly vary strength per image for natural diversity
        strength_var = args.strength + rng.uniform(-0.05, 0.05)
        strength_var = max(0.25, min(0.9, strength_var))

        ctx = (
            torch.autocast(device_type=device, dtype=dtype)
            if device == "cuda"
            else torch.no_grad()
        )
        with ctx:
            img = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                image=ref_img,
                strength=strength_var,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=g,
            ).images[0]

        img_index = current_count + i
        save_path = out_dir / f"chotpoti_{img_index}.png"
        img.save(save_path)

    print(f"\n✅ Done! {remaining} images saved to: {out_dir}")
    print("\n💡 Tips:")
    print("   • Too similar to originals? → increase --strength (try 0.7–0.8)")
    print("   • Looks too unlike chotpoti? → decrease --strength (try 0.45–0.55)")
    print("   • Train/test images look identical? → use different --seed values")


if __name__ == "__main__":
    main()