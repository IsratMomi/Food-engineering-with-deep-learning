# plot_training_log.py
import pandas as pd
import matplotlib.pyplot as plt

# Path to your log file
LOG_PATH = "./checkpoints/training_log.csv"

def plot_log(csv_path=LOG_PATH):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Convert numeric columns
    for col in ["epoch", "train_acc", "val_acc", "train_loss", "best_val"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Plot training vs validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_acc"], label="Train Acc", marker="o")
    plt.plot(df["epoch"], df["val_acc"], label="Val Acc", marker="s")
    plt.plot(df["epoch"], df["best_val"], label="Best Val (so far)", linestyle="--", alpha=0.7)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Optional: plot training loss separately
    if "train_loss" in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df["epoch"], df["train_loss"], label="Train Loss", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    plot_log()