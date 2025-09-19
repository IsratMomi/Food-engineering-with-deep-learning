import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

def plot_top_confusions(y_true, y_pred, class_names, top_n=10, save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # normalize
    
    # Find top-N confused pairs (off-diagonal values)
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], i, j))
    confusions = sorted(confusions, key=lambda x: x[0], reverse=True)[:top_n]
    
    # Extract rows/cols for those classes
    selected_classes = sorted(set([i for _, i, j in confusions] + [j for _, i, j in confusions]))
    cm_focus = cm_norm[np.ix_(selected_classes, selected_classes)]
    labels_focus = [class_names[i] for i in selected_classes]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_focus, display_labels=labels_focus)
    disp.plot(ax=ax, cmap="magma", xticks_rotation=45, colorbar=True, values_format=".2f")
    plt.title(f"Top-{top_n} Most Confused Food-101 Classes (Normalized)")
    plt.tight_layout()
    plt.savefig("confusion_matrix_top_confusions.png", dpi=200)
    print("✅ Saved: confusion_matrix_top_confusions.png")
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"✅ Saved: {save_path}")
    return fig
    
