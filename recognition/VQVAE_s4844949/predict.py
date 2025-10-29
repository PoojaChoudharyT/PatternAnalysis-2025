"""
Author: Pooja Choudhary
Student ID: 48449496

"""

from __future__ import annotations
import os
import argparse
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import get_dataloaders
from modules import VQVAE
from utils import get_device, set_seed, normalize_minmax_per_image, reconstruction_loss, ssim_per_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to HipMRI 2D slices root")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to trained checkpoint (.pt)")
    ap.add_argument("--outdir", type=str, default="Output/predict")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--num-samples", type=int, default=12, help="Number of sample panels to save")
    ap.add_argument("--grid-cols", type=int, default=6, help="Columns for the montage grid")
    ap.add_argument("--viz-codebook", action="store_true", help="Also save codebook usage figure")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Dataloaders
    _, _, test_loader = get_dataloaders(
        root_dir=args.data,
        batch_size=args.batch_size,
        normImage=False,
        categorical=False,
        target_size=(256, 128),
        num_workers=args.workers,
    )

    # Loading checkpoint and recreating model with saved hyperparams
    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = ckpt.get("args", {})


    model = VQVAE(
        in_channels=1,
        hidden_channels=ckpt_args.get("hid", ckpt_args.get("hidden_channels", 128)),
        res_hidden_channels=ckpt_args.get("res_hid", ckpt_args.get("res_hidden_channels", 64)),
        num_res_layers=ckpt_args.get("res_layers", ckpt_args.get("num_res_layers", 2)),
        embedding_dim=ckpt_args.get("emb_dim", ckpt_args.get("embedding_dim", 64)),
        num_embeddings=ckpt_args.get("n_codes", ckpt_args.get("num_embeddings", 512)),
        commitment_cost=ckpt_args.get("beta", ckpt_args.get("commitment_cost", 0.25)),
        ema_decay=ckpt_args.get("ema", ckpt_args.get("ema_decay", 0.99)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Evaluate
    os.makedirs(args.outdir, exist_ok=True)
    all_ssim = []
    saved = 0
    montage_tiles = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predict"):
            x = batch["image"].to(device)          
            x = normalize_minmax_per_image(x)

            z = model.encoder(x)
            z_e = model.pre_vq(z)
            z_q, vq_loss, perplexity, _, _ = model.vq(z_e)
            x_logits = model.decode(z_q)

            # get recon in [0,1]
            _, x_recon = reconstruction_loss(x_logits, x, kind="l1")

            # SSIM per image
            ssim_vals = ssim_per_image(x_recon, x).cpu().numpy().tolist()
            all_ssim.extend(ssim_vals)

            # Save sample images
            for b in range(x.size(0)):
                if saved >= args.num_samples:
                    break
                orig = x[b, 0].cpu().numpy()
                recon = x_recon[b, 0].cpu().numpy()

                # Save side-by-side images
                out_path = os.path.join(args.outdir, f"sample_{saved:03d}.png")
                save_panel(orig, recon, out_path)

                # Also prepare paired tiles for montage
                h, w = orig.shape
                pair = np.zeros((h * 2, w), dtype=np.float32)
                pair[:h, :] = orig
                pair[h:, :] = recon
                montage_tiles.append(pair)

                saved += 1
            if saved >= args.num_samples:
                break

    mean_ssim = float(np.mean(all_ssim)) if all_ssim else 0.0
    print(f"[Test] Mean SSIM over evaluated test batches: {mean_ssim:.4f}")
    with open(os.path.join(args.outdir, "predict_summary.json"), "w") as f:
        json.dump({"mean_test_ssim": mean_ssim, "num_samples": saved}, f, indent=2)


    print(f"Saved {saved} images to: {args.outdir}")
    print("Done.")




def save_panel(orig: np.ndarray, recon: np.ndarray, out_path: str):
    """Save side-by-side original vs reconstruction panel."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1); plt.imshow(orig, cmap="gray"); plt.title("Original"); plt.axis("off")
    plt.subplot(1, 2, 2); plt.imshow(recon, cmap="gray"); plt.title("Reconstruction"); plt.axis("off")
    plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()



if __name__ == "__main__":
    main()
