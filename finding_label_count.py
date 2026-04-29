from pathlib import Path

# Point to your Food101 train folder
train_dir = Path(r"E:\er\project-food\data\food-101\food-101\images")
classes = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
for i, c in enumerate(classes):
    print(f"{i:3}  {c}")