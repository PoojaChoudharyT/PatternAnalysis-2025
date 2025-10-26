"""
Author: Pooja Choudhary
Student ID: 48449496
""" 


# ==============================================
# HiP MRI Data Loader
# ==============================================
# This script provides tools to:
# - Load 2D MRI slices from NIfTI (.nii.gz) files
# - Filter images by target dimensions
# - Split dataset into train/validation/test sets safely
# - Normalize and transform data for PyTorch
# - Visualize sample images from the dataset
# ==============================================

#Importing necessary libraries
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torchvision.transforms import Compose, Grayscale, ToTensor
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt




# ==============================================
# Helper Functions
# ==============================================

def to_channels(arr: np.ndarray, dtype = np.uint8 ) -> np.ndarray:

    """
    Convert a 2D label image to a one-hot encoded channel representation.
    Each unique value in the array becomes a separate channel.

    Args:
        arr (np.ndarray): Input 2D array with categorical labels.
        dtype: Data type for the output array.

    Returns:
        np.ndarray: 3D array (H x W x num_channels) one-hot encoded.
    """
    
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels), ), dtype = dtype)
    for c in channels:
        c = int(c)
        res[..., c:c +1][arr == c] = 1
    
    return res


# load medical image functions
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
            inImage = utils.to_channels ( inImage , dtype = dtype )
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





# ==============================================
# 1. Dataset Shape Summary
# ==============================================
def summarize_split_shapes(root_dir):
    """
    Scan train/validate/test folders, count per-shape frequencies and print summary.

    Args:
        root_dir (str): Root folder containing dataset splits.

    Returns:
        Counter: Global shape counts across all splits.
    """
    split_folders = {
        "train": os.path.join(root_dir, "train"),
        "validate": os.path.join(root_dir, "validate"),
        "test": os.path.join(root_dir, "test"),
    }

    # Handle alternate folder names (e.g., keras_slices_train)
    for key, path in split_folders.items():
        if not os.path.isdir(path):
            alt = os.path.join(root_dir, f"keras_slices_{key}")
            if os.path.isdir(alt):
                split_folders[key] = alt

    global_counter = Counter()
    print("\n================ MRI DATASET SHAPE SUMMARY ================")

    for split, folder in split_folders.items():
        if not os.path.isdir(folder):
            print(f"[WARNING] Folder not found for split '{split}': {folder}")
            continue

        nii_files = sorted(glob.glob(os.path.join(folder, "**/*.nii.gz"), recursive=True))
        local_counter = Counter()

        print(f"\n[SCAN] Split: {split:9s} | Folder: {folder}")
        print(f"[SCAN] Files found: {len(nii_files)}")

        for fpath in tqdm(nii_files, desc=f"Reading {split} files"):
            try:
                arr = np.asarray(nib.load(fpath).get_fdata(), dtype=np.float32)
            except Exception as e:
                print(f"[ERROR] Failed to read {fpath}: {e}")
                continue

            # Remove singleton dimensions if present
            if arr.ndim == 3 and 1 in arr.shape:
                arr = np.squeeze(arr)

            if arr.ndim != 2:
                continue

            shape = arr.shape
            local_counter[shape] += 1
            global_counter[shape] += 1

        print("[SUMMARY] Per-shape counts for this split:")
        for (h, w), cnt in local_counter.most_common():
            print(f"  - {h} x {w} → {cnt} images")

    print("\n[GLOBAL] Combined shape counts across ALL splits:")
    for (h, w), cnt in global_counter.most_common():
        print(f"  - {h} x {w} → {cnt} images")

    print(f"[GLOBAL] Total valid 2D images counted: {sum(global_counter.values())}")
    print("===========================================================\n")
    return global_counter

# ==============================================
# 2. Utility Functions for File Handling
# ==============================================
def get_all_image_files(root_dir):
    """Recursively collect all .nii.gz files under a root folder."""
    image_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".nii.gz"):
                image_files.append(os.path.join(dirpath, f))
    return sorted(image_files)

def filter_image_files_by_dimension(image_files, target_size=(256, 128)):
    """
    Keep only images matching the target 2D shape.
    This ensures consistency for batch processing in PyTorch.
    """
    valid_images = []
    for path in tqdm(image_files, desc="Filtering by target size"):
        shape = nib.load(path).get_fdata(caching="unchanged").shape
        if len(shape) == 3:
            shape = shape[:2]
        if shape == target_size:
            valid_images.append(path)
    return valid_images

# ==============================================
# 3. PyTorch Dataset Class
# ==============================================
class ProstateMRIDataset(Dataset):
    """
    Dataset class for pre-split 2D MRI slices.

    Automatically:
        - Loads all images in memory
        - Filters by target dimensions
        - Normalizes or converts to categorical if needed
        - Provides PyTorch tensor access via __getitem__
    """

    def __init__(self, split_dir, normImage=False, categorical=False,
                 dtype=np.float32, target_size=(256, 128)):
        self.split_dir = split_dir
        self.normImage = normImage
        self.categorical = categorical
        self.dtype = dtype
        self.target_size = target_size

        # Get all files in the split folder
        all_image_files = get_all_image_files(split_dir)

        # Keep only those matching the target shape
        self.image_files = filter_image_files_by_dimension(
            all_image_files, target_size=target_size
        )

        # Preload images for fast access
        self.images = load_data_2D(
            self.image_files,
            normImage=self.normImage,
            categorical=self.categorical,
            dtype=self.dtype
        )

        # Transform pipeline to convert numpy arrays to PyTorch tensors
        self.transform = Compose([
            Grayscale(num_output_channels=1),
            ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Return a single sample as a dictionary:
            - 'image': PyTorch tensor
            - 'path': Original file path
        """
        image_np = self.images[idx]
        image_pil = Image.fromarray(image_np)
        image_tensor = self.transform(image_pil)
        return {"image": image_tensor, "path": self.image_files[idx]}

# ==============================================
# 4. DataLoader Builder
# ==============================================
def get_dataloaders(root_dir, batch_size=16, normImage=False, categorical=False,
                    target_size=(256, 128), num_workers=0):
    """
    Build PyTorch DataLoaders for train/validate/test splits.

    Automatically handles folders with standard names or keras_slices_* prefixes.
    """
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "validate")
    test_dir = os.path.join(root_dir, "test")

    if not os.path.isdir(train_dir):
        train_dir = os.path.join(root_dir, "keras_slices_train")
    if not os.path.isdir(val_dir):
        val_dir = os.path.join(root_dir, "keras_slices_validate")
    if not os.path.isdir(test_dir):
        test_dir = os.path.join(root_dir, "keras_slices_test")

    train_dataset = ProstateMRIDataset(train_dir, normImage, categorical, target_size=target_size)
    val_dataset = ProstateMRIDataset(val_dir, normImage, categorical, target_size=target_size)
    test_dataset = ProstateMRIDataset(test_dir, normImage, categorical, target_size=target_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader