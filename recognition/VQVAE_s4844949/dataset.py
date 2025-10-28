"""
Author: Pooja Choudhary
Student ID: 48449496


HiP MRI Data Loader

""" 




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
import matplotlib.pyplot as plt
from utils import load_data_2D



# 1. Dataset Shape Summary
def summarize_split_shapes(root_dir):
    
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


# 2. Utility Functions for File Handling

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


# 3.  ProstateMRI Dataset Class
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


# 4. DataLoader Builder

def get_dataloaders(root_dir, batch_size=16, normImage=False, categorical=False,
                    target_size=(256, 128), num_workers=0):
    
    #Build PyTorch DataLoaders for train/validate/test splits.

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



# 5. Visualization Utilities

def visualize_samples(dataloader, num_samples=5, save_path="Output/sample_visualization.png"):
    """
    Display a few preprocessed samples from a DataLoader.
    Saves the figure to disk for reference.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dataset = dataloader.dataset

    plt.figure(figsize=(12, 6))
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        img_tensor = sample["image"]

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img_tensor.squeeze(), cmap="gray")
        plt.title(f"Sample {i+1}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()

def visualize_original_vs_preprocessed(dataset, num_samples=5):
    """
    Compare original raw images with preprocessed images side by side.

    Useful for sanity-checking preprocessing, normalization, or resizing effects.

    Args:
        dataset: ProstateMRIDataset instance
        num_samples: Number of samples to visualize
    """
    plt.figure(figsize=(10, 4 * num_samples))
    
    for i in range(min(num_samples, len(dataset))):
        original_img = nib.load(dataset.image_files[i]).get_fdata()
        if original_img.ndim == 3:
            original_img = original_img[:, :, 0]

        preprocessed_img = dataset.images[i]

        plt.subplot(num_samples, 2, 2*i + 1)
        plt.imshow(original_img, cmap='gray')
        plt.title(f"Original Sample {i+1}")
        plt.axis('off')

        plt.subplot(num_samples, 2, 2*i + 2)
        plt.imshow(preprocessed_img, cmap='gray')
        plt.title(f"Preprocessed Sample {i+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()