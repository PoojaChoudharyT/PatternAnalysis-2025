"""
Author: Pooja Choudhary
Student ID: 48449496

Train the VQVAE model
"""


from __future__ import annotations
import torch
import os
import json
import argparse
from modules import VQVAE
from dataset import get_dataloaders
from typing import Tuple
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from utils import reconstruction_loss, normalize_minmax_per_image, set_seed, ssim_per_image, save_checkpoint
from visualization_utils import plot_curves


# Warm-start codebook from data
@torch.no_grad()
def warm_start_codebook(model: VQVAE, loader, device, max_batches: int = 4):
    
    #Initializes the VQ codebook using real encoder latents from a few mini-batches.
    #This prevents early collapse to a single code.
    
    model.eval()
    collected = []
    count = 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        x = normalize_minmax_per_image(x)  # ensuring [0,1]
        # encode to get z_e
        z = model.encoder(x)
        z_e = model.pre_vq(z)                      
        collected.append(z_e.detach())
        count += 1
        if count >= max_batches:
            break
    if collected:
        z_e_all = torch.cat(collected, dim=0)
        model.vq.init_from_data(z_e_all)
    model.train()



# Diversity regularizer
def diversity_loss_from_encodings(encodings: torch.Tensor, num_codes: int, w: float) -> torch.Tensor:
    """
    Gentle push toward uniform code usage.
    encodings: [N, K] one-hot from the VQ forward
    loss = w * KL(p || uniform) = w * sum p * log(p * K)
    """
    if w <= 0.0:
        return torch.tensor(0.0, device=encodings.device, dtype=encodings.dtype)
    avg_probs = encodings.mean(dim=0)  # [K]
    avg_probs = torch.clamp(avg_probs, min=1e-6)
    kl_uniform = torch.sum(avg_probs * torch.log(avg_probs * num_codes))
    return w * kl_uniform



# Training / Evaluation loops
# (explicit encode/quantize/decode to access z_e & encodings)

def train_one_epoch(
    model: VQVAE, loader, optimizer, device,
    epoch: int, explore_epochs: int, base_noise_std: float,
    diversity_weight: float, recon_kind: str = "l1"
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    sum_ssim = 0.0
    total_ppl = 0.0
    n_batches = 0
    n_images = 0

    # noise std decays to 0 across explore_epochs
    if explore_epochs > 0:
        t = max(0, explore_epochs - epoch + 1) / max(1, explore_epochs)
        noise_std = base_noise_std * t
    else:
        noise_std = 0.0

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        x = batch["image"].to(device, non_blocking=True)
        x = normalize_minmax_per_image(x)  

        # explicit pass so we can inject noise & get encodings
        z = model.encoder(x)            
        z_e = model.pre_vq(z)          

        if noise_std > 0.0:
            z_e = z_e + noise_std * torch.randn_like(z_e)

        z_q, vq_loss, perplexity, encodings, _ = model.vq(z_e)  
        x_logits = model.decode(z_q)
        recon_loss, x_recon = reconstruction_loss(x_logits, x, kind=recon_kind)

        # diversity regularizer 
        div_loss = diversity_loss_from_encodings(encodings, model.vq.num_embeddings, diversity_weight)

        loss = recon_loss + vq_loss + div_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        with torch.no_grad():
            ssims = ssim_per_image(x, x_recon)

        total_loss += loss.item()
        sum_ssim += ssims.sum().item()
        total_ppl += float(perplexity)
        n_batches += 1
        n_images += ssims.numel()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            ssim=f"{float(ssims.mean().item()):.3f}",
            ppl=f"{float(perplexity):.2f}",
            noise=f"{noise_std:.3f}"
        )

    avg_loss = total_loss / max(n_batches, 1)
    avg_ssim = sum_ssim / max(n_images, 1)
    avg_ppl = total_ppl / max(n_batches, 1)
    return avg_loss, avg_ssim, avg_ppl


@torch.no_grad()
def evaluate(
    model: VQVAE, loader, device, recon_kind: str = "l1"
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    sum_ssim = 0.0
    total_ppl = 0.0
    n_batches = 0
    n_images = 0

    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["image"].to(device, non_blocking=True)
        x = normalize_minmax_per_image(x)

        z = model.encoder(x)
        z_e = model.pre_vq(z)
        z_q, vq_loss, perplexity, _, _ = model.vq(z_e)
        x_logits = model.decode(z_q)
        recon_loss, x_recon = reconstruction_loss(x_logits, x, kind=recon_kind)

        loss = recon_loss + vq_loss
        ssims = ssim_per_image(x, x_recon) 

        total_loss += loss.item()
        sum_ssim += ssims.sum().item()
        total_ppl += float(perplexity)
        n_batches += 1
        n_images += ssims.numel()

    avg_loss = total_loss / max(n_batches, 1)
    avg_ssim = sum_ssim / max(n_images, 1)
    avg_ppl = total_ppl / max(n_batches, 1)
    return avg_loss, avg_ssim, avg_ppl



# Main
def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # Data: turn OFF internal normalization; we normalize to [0,1] here
    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=args.data,
        batch_size=args.batch_size,
        normImage=False,            # important for consistency with sigmoid loss
        categorical=False,
        target_size=(256, 128),
        num_workers=args.workers,
    )

    # Model
    model = VQVAE(
        in_channels=1,
        hidden_channels=args.hid,
        res_hidden_channels=args.res_hid,
        num_res_layers=args.res_layers,
        embedding_dim=args.emb_dim,
        num_embeddings=args.n_codes,
        commitment_cost=args.beta,
        ema_decay=args.ema,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # LR schedule: warmup + cosine
    total_steps_per_epoch = max(1, len(train_loader))
    warmup_steps = max(1, int(args.warmup_frac * args.epochs * total_steps_per_epoch))
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs * total_steps_per_epoch - warmup_steps))

    def step_scheduler(global_step):
        if global_step < warmup_steps:
            # Linear warmup from 10% to 100% of lr
            warmup_ratio = 0.1 + 0.9 * (global_step / max(1, warmup_steps))
            for g in optimizer.param_groups:
                g["lr"] = args.lr * warmup_ratio
        else:
            cosine.step()

    # Warm-start codebook from a few batches
    if args.warm_init_batches > 0:
        print(f"Warm-starting codebook from {args.warm_init_batches} batch(es)...")
        warm_start_codebook(model, train_loader, device, max_batches=args.warm_init_batches)

    history = {
        "train_loss": [], "val_loss": [],
        "train_ssim": [], "val_ssim": [],
        "train_ppl": [],  "val_ppl":  [],
    }

    os.makedirs(args.outdir, exist_ok=True)
    best_val_ssim = -1.0
    patience_counter = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        tr_loss, tr_ssim, tr_ppl = train_one_epoch(
            model, train_loader, optimizer, device,
            epoch=epoch,
            explore_epochs=args.explore_epochs,
            base_noise_std=args.noise_std,
            diversity_weight=args.div_weight,
            recon_kind=args.recon_kind,
        )
        va_loss, va_ssim, va_ppl = evaluate(model, val_loader, device, recon_kind=args.recon_kind)

        history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
        history["train_ssim"].append(tr_ssim); history["val_ssim"].append(va_ssim)
        history["train_ppl"].append(tr_ppl);   history["val_ppl"].append(va_ppl)

        print(
            f"train: loss={tr_loss:.4f}  ssim={tr_ssim:.3f}  ppl={tr_ppl:.2f} | "
            f"val: loss={va_loss:.4f}  ssim={va_ssim:.3f}  ppl={va_ppl:.2f}"
        )

        # Save best by val SSIM
        if va_ssim > best_val_ssim:
            best_val_ssim = va_ssim
            patience_counter = 0
            ckpt_path = save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "args": vars(args),
                    "val_ssim": va_ssim,
                },
                args.outdir,
                tag=f"best_ssim_{va_ssim:.3f}",
            )
            print(f"Saved new best checkpoint → {ckpt_path}")
        else:
            if args.patience > 0:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered at epoch {epoch} — validation SSIM plateaued.")
                    break

        # Plot curves each epoch
        plot_curves(history, args.outdir)

        # Step LR scheduler across the epoch's steps (approximate: 1 step/it)
        global_step += len(train_loader)
        step_scheduler(global_step)

    # Final test eval on best checkpoint
    best_path = None
    for f in sorted(os.listdir(args.outdir)):
        if f.startswith("checkpoint_best_ssim_") and f.endswith(".pt"):
            best_path = os.path.join(args.outdir, f)
    if best_path:
        print(f"\nLoading best checkpoint for test: {best_path}")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    te_loss, te_ssim, te_ppl = evaluate(model, test_loader, device, recon_kind=args.recon_kind)
    print(f"\nTEST: loss={te_loss:.4f}  ssim={te_ssim:.3f}  ppl={te_ppl:.2f}")

    # Save summary JSON 
    summary = {
        "best_val_ssim": best_val_ssim,
        "final_test_loss": te_loss,
        "final_test_ssim": te_ssim,
        "final_test_ppl": te_ppl,
        "history": history,
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to HipMRI 2D slices root")
    ap.add_argument("--outdir", type=str, default="runs/hipmri_vqvae")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--res-hid", type=int, default=64)
    ap.add_argument("--res-layers", type=int, default=2)
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--n-codes", type=int, default=512)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--warm-init-batches", type=int, default=4, help="Batches to warm-start codebook")
    ap.add_argument("--explore-epochs", type=int, default=5, help="Apply encoder-noise for first N epochs")
    ap.add_argument("--noise-std", type=float, default=0.2, help="Std of Gaussian noise on z_e at epoch 1 (decays to 0)")
    ap.add_argument("--div-weight", type=float, default=5e-3, help="Weight for diversity (uniform-KL) regularizer")
    ap.add_argument("--recon-kind", type=str, default="l1", choices=["l1", "bce"])
    ap.add_argument("--warmup-frac", type=float, default=0.1, help="Fraction of total steps for LR warmup")
    ap.add_argument("--patience", type=int, default=0, help="Early stopping patience in epochs (0=disabled)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    main(args)
