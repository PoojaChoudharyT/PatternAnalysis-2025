"""
Author: Pooja Choudhary
Student ID: 48449496

"""


import torch
import torch.nn as nn
import torch.nn.functional as F



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Residual Blocks

class ResidualLayer(nn.Module):
    """
    Single 3x3 -> ReLU -> 1x1 residual layer with skip connection.
    """
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)

class ResidualStack(nn.Module):
    """
    A stack of ResidualLayer modules applied sequentially, ending with ReLU.
    """
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [ResidualLayer(in_channels, hidden_channels) for _ in range(num_layers)]
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return F.relu(x, inplace=True)



# Encoder

class Encoder(nn.Module):
    """
    Encodes input MRI images into latent space before quantization.
    Output shape: [B, embedding_dim, H/4, W/4]

    Downsampling encoder for MRI slices.

    Reduces spatial size by 4× using two strided convolutions:
      (H, W) -> (H/2, W/2) -> (H/4, W/4)

    For 256×128 input → 64×32 latent feature map.
    """
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 128,
        res_hidden_channels: int = 64,
        num_res_layers: int = 2,
    ):
        super().__init__()
        hc = hidden_channels
        self.net = nn.Sequential(
            # ×2 downsample
            nn.Conv2d(in_channels, hc // 2, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            # ×4 downsample total
            nn.Conv2d(hc // 2, hc, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            # bottleneck mixing
            nn.Conv2d(hc, hc, kernel_size=3, stride=1, padding=1, bias=True),
            ResidualStack(hc, res_hidden_channels, num_res_layers),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Vector Quantizer EMA (Codebook)

class VectorQuantizerEMA(nn.Module):
    """
    VQ layer using Exponential Moving Average updates.
    Maps continuous latent vectors to discrete codebook entries.

    
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = commitment_cost
        self.decay = decay
        self.eps = eps

        # Codebook: [D, K]
        embed = torch.randn(embedding_dim, num_embeddings)
        self.register_buffer("embedding", embed)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", embed.clone())


    @torch.no_grad()
    def _ema_update(self, flat_inputs: torch.Tensor, encodings: torch.Tensor):
        """
        EMA update of codebook statistics.
        flat_inputs: [N, D], encodings (one-hot): [N, K]
        """
        # Accumulate counts and sums
        cluster_size = encodings.sum(0)                           # [K]
        embed_sum = flat_inputs.t() @ encodings                   # [D, K]

        # Decay-accumulate
        self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

        # Laplace smoothing of counts to avoid zeros
        n = self.cluster_size.sum()
        smoothed_cluster_size = (self.cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n

        # Normalize to get actual embeddings
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(0)  # [D,K]
        self.embedding.copy_(embed_normalized)


    def forward(self, z_e):
        """
        Args:
            z_e: Latent encoder output [B, D, H, W]
        Returns:
            z_q: quantized latents [B, D, H, W]
            vq_loss: commitment loss (EMA handles codebook update)
            encodings: one-hot assignments [B*H*W, K]
            indices: code indices [B, H, W]
            perplexity: Codebook usage metric  (Perplexity is a measures which says how many codebook entries are actually being used)
        """
        B, D, H, W = z_e.shape
        assert D == self.embedding_dim, "Encoder channels must equal embedding_dim"

        # Flatten to [N, D]
        flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D)  # [N, D], N = B*H*W

        # Distances to codebook entries (use ||x||^2 + ||e||^2 - 2x·e)
        e = self.embedding  # [D, K]
        dist = (
            flat.pow(2).sum(1, keepdim=True)                      # [N,1]
            + e.pow(2).sum(0, keepdim=True)                       # [1,K]
            - 2 * (flat @ e)                                      # [N,K]
        )

        # Nearest code for each vector
        indices = torch.argmin(dist, dim=1)                       # [N]
        encodings = F.one_hot(indices, num_classes=self.num_embeddings).type(flat.dtype)  # [N,K]

        # Quantize
        z_q_flat = encodings @ e.t()                # [N,D]
        z_q = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()  # [B,D,H,W]

        # EMA updates (no gradients)
        if self.training:
            self._ema_update(flat.detach(), encodings.detach())

        # Commitment loss (codebook moved by EMA)
        vq_loss = self.beta * F.mse_loss(z_e.detach(), z_q)

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()

        # Perplexity
        avg_probs = encodings.mean(dim=0)     # [K]
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Reshape indices for convenience
        indices = indices.view(B, H, W)

        return z_q, vq_loss, perplexity, encodings, indices
    

    @torch.no_grad()
    def init_from_data(self, z_e: torch.Tensor):
        """
        Warm-start the codebook from real encoder outputs.
        z_e: [B, D, H, W]
        """
        B, D, H, W = z_e.shape
        assert D == self.embedding_dim, f"Expected D={self.embedding_dim}, got {D}"
        flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D)  # [N, D]
        N = flat.size(0)
        # If N < K, repeat to have enough samples
        if N < self.num_embeddings:
            reps = (self.num_embeddings + N - 1) // N
            flat = flat.repeat(reps, 1)

        # Pick K random vectors from data as initial codebook
        idx = torch.randperm(flat.size(0), device=flat.device)[: self.num_embeddings]
        chosen = flat[idx].t().contiguous()  # [D, K]

        # Initialize EMA buffers
        self.embedding.copy_(chosen)
        self.embed_avg.copy_(chosen)
        self.cluster_size.copy_(torch.ones(self.num_embeddings, device=flat.device))



# Decoder

class Decoder(nn.Module):
    """
    Reconstructs MRI slice from quantized latent vectors.

    Upsampling decoder for MRI slices.

    Two ConvTranspose2d layers upsample spatially by 4× in total to
    recover the input resolution (e.g., 64×32 → 128×64 → 256×128).

    Args:
        out_channels:          output channels (1 for grayscale)
        hidden_channels:       base channel width (match Encoder hidden)
        res_hidden_channels:   hidden width in residual layers
        num_res_layers:        number of residual layers in the stack
    """
    def __init__(
            self,
        out_channels: int = 1,
        hidden_channels: int = 128,
        res_hidden_channels: int = 64,
        num_res_layers: int = 2,
    ):
        super().__init__()
        hc = hidden_channels
        self.net = nn.Sequential(
            # bottleneck mixing
            nn.Conv2d(hc, hc, kernel_size=3, stride=1, padding=1, bias=True),
            ResidualStack(hc, res_hidden_channels, num_res_layers),

            # upsample ×2
            nn.ConvTranspose2d(hc, hc // 2, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),

            # upsample ×2 (total ×4) → logits
            nn.ConvTranspose2d(hc // 2, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
            # Note: outputs are logits; apply sigmoid in the loss for [0,1] data
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)



# VQ-VAE (Main Model)

class VQVAE(nn.Module):
    """
    Single-level VQ-VAE architecture for 2D prostate MRI slices.
    Pipeline:
        Encoder → 1x1 conv (to embedding_dim) → VectorQuantizerEMA
               → 1x1 conv (to hidden_channels) → Decoder
    """
    def __init__(
        self,
        in_channels=1,
        hidden_channels=128,
        res_hidden_channels=64,
        num_res_layers=2,
        embedding_dim=64,
        num_embeddings=512,
        commitment_cost=0.25,
        ema_decay=0.99,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            res_hidden_channels=res_hidden_channels,
            num_res_layers=num_res_layers,
        )
        # project encoder width -> embedding_dim (channels)
        self.pre_vq = nn.Conv2d(hidden_channels, embedding_dim, kernel_size=1, bias=True)

        # EMA codebook
        self.vq = VectorQuantizerEMA(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            decay=ema_decay,
        )

        # project quantized latents back to decoder width
        self.post_vq = nn.Conv2d(embedding_dim, hidden_channels, kernel_size=1, bias=True)

        self.decoder = Decoder(
            out_channels=in_channels,
            hidden_channels=hidden_channels,
            res_hidden_channels=res_hidden_channels,
            num_res_layers=num_res_layers,
        )


    def encode(self, x: torch.Tensor):
        """
        Returns:
            z_e: pre-quantization latents [B, D, H', W']
            z_q: quantized latents [B, D, H', W']
            vq_loss: commitment loss (scalar)
            perplexity: codebook usage
            indices: code indices [B, H', W']
        """
        z = self.encoder(x)                 # [B, hidden, H', W']
        z_e = self.pre_vq(z)                # [B, emb_dim, H', W']
        z_q, vq_loss, perplexity, _, idx = self.vq(z_e)
        return z_e, z_q, vq_loss, perplexity, idx


    def decode(self, z_q: torch.Tensor):
        z_q_up = self.post_vq(z_q)          # [B, hidden, H', W']
        x_logits = self.decoder(z_q_up)     # logits
        return x_logits


    def forward(self, x: torch.Tensor, recon_loss_type: str ="l1"):
        """
        Forward pass through VQ-VAE.
        x: [B,1,H,W] in [0,1]
        recon_loss_type: "l1" or "bce"
        Returns:
            dict containing:
              - loss_total
              - loss_recon
              - loss_vq
              - perplexity
              - x_recon_logits, x_recon (sigmoid), z_q
        """
        z_e, z_q, vq_loss, perplexity, _ = self.encode(x)
        x_recon_logits = self.decode(z_q)

        if recon_loss_type.lower() == "bce":
            loss_recon = F.binary_cross_entropy_with_logits(x_recon_logits, x)
            x_recon = torch.sigmoid(x_recon_logits)
        else:
            x_recon = torch.sigmoid(x_recon_logits)
            loss_recon = F.l1_loss(x_recon, x)

        loss_total = loss_recon + vq_loss

        return {
            "loss_total": loss_total,
            "loss_recon": loss_recon,
            "loss_vq": vq_loss,
            "perplexity": perplexity,
            "x_recon_logits": x_recon_logits,
            "x_recon": x_recon,
            "z_q": z_q,
        }