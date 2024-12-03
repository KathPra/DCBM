import os
import shutil
import random
from pathlib import Path

def create_class_balanced_subset(original_dir, subset_dir, ratio=0.1):
    """
    Creates a random, class-balanced subset of the dataset.
    
    Parameters:
        original_dir (str): Path to the original dataset directory (organized by class).
        subset_dir (str): Path to the new subset directory.
        ratio (float): Proportion of the dataset to move to the subset (default: 0.1).
    """
    original_dir = Path(original_dir)
    subset_dir = Path(subset_dir)

    if not original_dir.is_dir():
        raise ValueError(f"The provided path '{original_dir}' is not a directory.")
    
    if subset_dir.exists():
        raise ValueError(f"The provided path '{subset_dir}' already exists.")

    # Ensure the subset directory exists
    subset_dir.mkdir(parents=True, exist_ok=True)

    for class_dir in original_dir.iterdir():
        if not class_dir.is_dir():
            continue  # Skip non-directory files
        
        class_name = class_dir.name
        class_subset_dir = subset_dir / class_name
        class_subset_dir.mkdir(parents=True, exist_ok=True)

        # Get all files in the class directory
        all_files = list(class_dir.glob("*"))
        if not all_files:
            print(f"No files found in class directory '{class_name}', skipping.")
            continue
        
        # Shuffle and split the files
        random.shuffle(all_files)
        subset_size = max(1, int(len(all_files) * ratio))
        subset_files = all_files[:subset_size]
        remaining_files = all_files[subset_size:]

        # Copy subset files to the new directory
        for file in subset_files:
            shutil.copy(str(file), str(class_subset_dir / file.name))
        
        # Remove subset files from the original directory
        for file in subset_files:
            file.unlink()  # Deletes the file
            
        print(f"Processed class '{class_name}': Moved {len(subset_files)} to subset, kept {len(remaining_files)} in original.")

if __name__ == "__main__":
    # Paths to original dataset and subset directory
    original_dataset_path = "path/to/original_dataset" # e.g. ../cifar100/train
    subset_dataset_path = "path/to/subset_dataset"  # e.g. ../cifar100/val

    # Create the subset
    create_class_balanced_subset(original_dataset_path, subset_dataset_path, ratio=0.1)
