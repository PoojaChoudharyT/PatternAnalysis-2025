"""
Author: Pooja Choudhary
Student ID: 48449496


Utility helpers for HipMRI 2D pipelines: dataset, modules, train and predict

"""


import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import random
from tqdm import tqdm
from typing import Dict
from __future__ import annotations

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_channels(arr: np.ndarray, dtype = np.uint8 ) -> np.ndarray:

  
    #Convert a 2D label image to a one-hot encoded channel representation.
    #Each unique value in the array becomes a separate channel.

    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels), ), dtype = dtype)
    for c in channels:
        c = int(c)
        res[..., c:c +1][arr == c] = 1
    
    return res


def load_data_2D(imageNames , normImage = False, categorical = False, dtype = np.float32, getAffines = False, early_stop = False):
    '''
    Load medical image data from names , cases list provided into a list for each.
    This function pre - allocates 4D arrays for conv2d to avoid excessive memory usage.
    normImage : bool(normalise the image 0.0 -1.0)
    early_stop : Stop loading pre - maturely , leaves arrays mostly empty , for quick loading and testing scripts.
    '''
    
    affines = []
    
    #get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching = 'unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0] # sometimes extra dims , remove
    if categorical:
        first_case = to_channels(first_case, dtype = dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype = dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype = dtype)

    for i, inName in enumerate(tqdm(imageNames, desc ="Loading the images")):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching ='unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0] # sometimes extra dims in HipMRI_study data
            inImage = inImage.astype(dtype)
        if normImage:
            #~ inImage = inImage / np. linalg . norm ( inImage )
            #~ inImage = 255. * inImage / inImage .max ()
            inImage = (inImage - inImage.mean()) /inImage.std()
        if categorical:
            inImage = to_channels ( inImage , dtype = dtype )
            images [i, :, :, :] = inImage
        else :
            images [i, :, :] = inImage
            
        affines.append(affine)
        if i > 20 and early_stop :
            break
            
    if getAffines:
        return images, affines
    else:
        return images



# Reconstruction loss helper
def reconstruction_loss(
    x_recon_logits: torch.Tensor, x: torch.Tensor, kind: str = "l1"
):
    """
    Return (loss, x_recon) given decoder logits and target x.

    kind:
        - 'l1'  (default) uses L1 on sigmoid(x_recon_logits)
        - 'bce' uses BCEWithLogits on logits (also returns sigmoid for viewing)
    """
    kind = (kind or "l1").lower()
    if kind == "bce":
        loss = F.binary_cross_entropy_with_logits(x_recon_logits, x)
        x_recon = torch.sigmoid(x_recon_logits)
    else:
        x_recon = torch.sigmoid(x_recon_logits)
        loss = F.l1_loss(x_recon, x)
    return loss, x_recon


#For reproducability
def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



# Normalize per-image min-max to [0,1]
def normalize_minmax_per_image(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    b = x.size(0)
    x_flat = x.view(b, -1)
    mins = x_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    maxs = x_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    return (x - mins) / (maxs - mins + eps)


#To calculate ssim per image
def ssim_per_image(
    x: torch.Tensor, y: torch.Tensor, C1: float = 0.01 ** 2, C2: float = 0.03 ** 2
) -> torch.Tensor:
    """Per-image SSIM for tensors in [0,1]; returns (B,)"""
    x = torch.clamp(x, 0, 1)
    y = torch.clamp(y, 0, 1)
    mu_x = F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)
    mu_y = F.avg_pool2d(y, kernel_size=7, stride=1, padding=3)
    sigma_x = F.avg_pool2d(x * x, 7, 1, 3) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 7, 1, 3) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 7, 1, 3) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    )
    return ssim_map.flatten(2).mean(dim=2).squeeze()


#To save checkpoints when model is training
def save_checkpoint(state: Dict, outdir: str, tag: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"checkpoint_{tag}.pt")
    torch.save(state, path)
    return path