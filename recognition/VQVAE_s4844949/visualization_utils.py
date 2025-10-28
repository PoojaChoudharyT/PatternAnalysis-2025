"""
Author: Pooja Choudhary
Student ID: 48449496

"""
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_curves(history: dict, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    #Defining custom colors
    colors = {
        "train": "#800080",  # purple
        "val": "#FF0000",    # red
        "hist": "#2ca02c"    # green
    }

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    plt.style.use("default")

    # Loss 
    ax = axs[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss", color=colors["train"], linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Val Loss", color=colors["val"], linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss", fontsize=13)
    y_all = history["train_loss"] + history["val_loss"]
    ax.set_ylim(min(y_all) * 0.9, max(y_all) * 1.1)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)


    # ssim
    ax = axs[1]
    ax.plot(epochs, history["train_ssim"], label="Train SSIM", color=colors["train"], linewidth=2)
    ax.plot(epochs, history["val_ssim"], label="Val SSIM", color=colors["val"], linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("SSIM"); ax.set_title("SSIM", fontsize=13)
    y_all = history["train_ssim"] + history["val_ssim"]
    ax.set_ylim(min(y_all) - 0.05, max(y_all) + 0.02)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)


    # Perplexity
    ax = axs[2]
    ax.plot(epochs, history["train_ppl"], label="Train PPL", color=colors["train"], linewidth=2)
    ax.plot(epochs, history["val_ppl"], label="Val PPL", color=colors["val"], linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Perplexity"); ax.set_title("Codebook Perplexity", fontsize=13)
    y_all = history["train_ppl"] + history["val_ppl"]
    ax.set_ylim(min(y_all) * 0.9, max(y_all) * 1.1)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "training_curves_pretty.png"), dpi=220, bbox_inches="tight")
    plt.close()