import os
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

# ------------------- Prompt Helpers -------------------
def to_prompt(label: str) -> str:
    """Builds a descriptive prompt for a food class name."""
    text = label.replace("_", " ")
    return (
        f"high quality food photo of {text}, professional food photography, "
        f"studio lighting, appetizing, detailed, realistic"
    )

NEGATIVE_PROMPT = (
    "blurry, low quality, text, watermark, logo, deformed, unrealistic, oversaturated"
)

# ------------------- Main -------------------
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic food images")
    parser.add_argument("--class-dir", type=str,
                        default=r"E:\er\project-food\data\food-101\images",
                        help="Path to Food-101 images directory")
    parser.add_argument("--out-root", type=str,
                        default=r"E:\er\project-food\data\synthetic_data",
                        help="Output directory for synthetic images")
    parser.add_argument("--per-class", type=int, default=20,
                        help="How many images to generate per class")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Stable Diffusion model ID")
    parser.add_argument("--steps", type=int, default=20, help="Diffusion steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")

    args = parser.parse_args()

    # Collect class names
    class_dir = Path(args.class_dir)
    classes = [d.name for d in class_dir.iterdir() if d.is_dir()]
    print(f"✅ Found {len(classes)} classes under {class_dir}")

    # Prepare output
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load Stable Diffusion
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"🚀 Loading {args.model} on {device} …")

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    # Generate per class
    for cls in classes:
        out_dir = out_root / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"🍽 Generating {args.per_class} images for class: {cls}")
        for i in tqdm(range(args.per_class), desc=cls):
            prompt = to_prompt(cls)
            g = torch.Generator(device=device).manual_seed(args.seed + i)

            with torch.autocast(device_type=device, dtype=dtype) if device == "cuda" else torch.no_grad():
                img = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    generator=g,
                ).images[0]

            out_path = out_dir / f"{cls}_{i}.png"
            img.save(out_path)

    print(f"✅ Done. Synthetic images saved under {out_root}")


if __name__ == "__main__":
    main()