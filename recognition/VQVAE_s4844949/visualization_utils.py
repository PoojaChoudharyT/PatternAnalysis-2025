"""
Author: Pooja Choudhary
Student ID: 48449496

"""
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from utils import normalize_minmax_per_image


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




@torch.no_grad()
def visualize_codebook_usage(
    model, loader, device,
    save_path: str = "Output/codebook_usage.png",
    sample_index: int = 0,
    cmap_latent: str = "viridis",
    cmap_quant: str = "viridis",
    cmap_indices: str = "tab20"
):
   
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()

    batch = next(iter(loader))
    x = batch["image"].to(device)
    x = normalize_minmax_per_image(x)

    # Forward
    z = model.encoder(x)         
    z_e = model.pre_vq(z)          
    z_q, vq_loss, perplexity, encodings, indices = model.vq(z_e) 

    b = sample_index
    assert 0 <= b < z_e.size(0), f"sample_index {b} out of range (batch size {z_e.size(0)})"

    # Choosing the most "interesting" channel (highest variance)
    variances = z_e[b].var(dim=(1, 2))
    c = int(torch.argmax(variances).item())

    ze = z_e[b, c].detach().cpu().float().numpy()
    zq = z_q[b, c].detach().cpu().float().numpy()
    idxmap = indices[b].detach().cpu().numpy()

    # Code usage counts (over the whole batch for stability)
    counts = encodings.sum(dim=0).detach().cpu().numpy() 

    fig = plt.figure(figsize=(8, 10))
    fig.suptitle("Latent & Quantized Spaces + Codebook Usage", fontsize=14, y=0.98)

    # Latent channel heatmap
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    im1 = ax1.imshow(ze, cmap=cmap_latent, aspect="auto")
    ax1.set_title("Latent Space (z_e) — selected channel", fontsize=12)
    ax1.set_xlabel("W'"); ax1.set_ylabel("H'")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)

    # Quantized channel heatmap
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    im2 = ax2.imshow(zq, cmap=cmap_quant, aspect="auto")
    ax2.set_title("Quantized Space (z_q) - same channel", fontsize=12)
    ax2.set_xlabel("W'"); ax2.set_ylabel("H'")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)

    # Codeblocks indices map
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    im3 = ax3.imshow(idxmap, cmap=cmap_indices, aspect="auto", interpolation="nearest")
    ax3.set_title("Codebook Indices Map", fontsize=12)
    ax3.set_xlabel("W'"); ax3.set_ylabel("H'")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.02)

    # Codeblocks usage histogram
    ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    ax4.bar(np.arange(len(counts)), counts, color="#800080", alpha=0.8, width=1.0)
    ax4.set_title(f"Codebook Usage Histogram (Perplexity: {float(perplexity):.1f})", fontsize=12)
    ax4.set_xlabel("Codebook ID"); ax4.set_ylabel("Count")
    ax4.set_xlim(0, len(counts) - 1)
    ax4.grid(alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"[Saved] Codebook usage figure: {save_path}")
    print(f"Selected channel: {c}; Latent shape: {ze.shape}; Index map shape: {idxmap.shape}")