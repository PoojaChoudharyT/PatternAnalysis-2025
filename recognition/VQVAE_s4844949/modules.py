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
    """
    def __init__(self, in_channels, hidden_channels, res_hidden_channels, num_res_layers):
        super().__init__()
        # TODO: define encoder convolution blocks + residual stack


    def forward(self, x):
        # TODO: pass input through encoder layers
        return x


# Vector Quantizer EMA (Codebook)

class VectorQuantizerEMA(nn.Module):
    """
    VQ layer using Exponential Moving Average updates.
    Maps continuous latent vectors to discrete codebook entries.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99):
        super().__init__()
        # TODO: initialize codebook variables


    def forward(self, z_e):
        """
        Args:
            z_e: Latent encoder output [B, D, H, W]
        Returns:
            z_q: Quantized tensor
            vq_loss: Commitment loss
            perplexity: Codebook usage metric
        """
        # TODO: implement nearest code lookup and EMA update
        return z_e, torch.tensor(0.), torch.tensor(0.), None, None



# Decoder

class Decoder(nn.Module):
    """
    Reconstructs MRI slice from quantized latent vectors.
    """
    def __init__(self, out_channels, hidden_channels, res_hidden_channels, num_res_layers):
        super().__init__()
        # TODO: define transpose-conv decoding path + residual stack


    def forward(self, z):
        # TODO: forward decode
        return z



# VQ-VAE (Main Model)

class VQVAE(nn.Module):
    """
    Single-level VQ-VAE architecture for 2D prostate MRI slices.
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

        # TODO: instantiate encoder, pre-VQ 1x1 conv, VQ, post-VQ conv, and decoder


    def encode(self, x):
        # TODO: encode + quantize
        return None


    def decode(self, z_q):
        # TODO: decode back to image
        return None


    def forward(self, x, recon_loss_type="l1"):
        """
        Forward pass through VQ-VAE.
        Returns:
            dict containing:
              - loss_total
              - loss_recon
              - loss_vq
              - perplexity
              - x_recon
        """
        # TODO: full forward pass
        return {}
