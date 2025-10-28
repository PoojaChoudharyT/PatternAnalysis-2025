"""
Author: Pooja Choudhary
Student ID: 48449496

"""

from __future__ import annotations
import argparse
import numpy as np
import torch
from dataset import get_dataloaders
from utils import get_device


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

if __name__ == "__main__":
    main()
