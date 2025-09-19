# download_food101.py
from torchvision.datasets import Food101
from torchvision import transforms
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def download_dataset():
    # Root directory to store the dataset
    root = "./data"

    # Define optional transforms (not required for download)
    basic_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
    ])

    # Download train and test splits
    train_ds = Food101(root=root, split="train", download=True, transform=basic_transform)
    test_ds = Food101(root=root, split="test", download=True, transform=basic_transform)

    print("✅ Food-101 dataset downloaded and initialized.")

# Run the function
if __name__ == "__main__":
    download_dataset()