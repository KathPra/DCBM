## conda env: semsam

## import packages
import numpy as np
import os
import torch
import torch.multiprocessing as mp
import glob
import tqdm
import argparse

from utilities.Semantic_SAM.semantic_sam import build_semantic_sam, SemanticSamAutomaticMaskGenerator
from utilities.data_loading import load_data
from utilities.SemSAM import generate_mask

torch.cuda.empty_cache() 

# Define multiprocessing worker
def main(dataset, level, create_segments, create_stats, device, check_existing):
    # load data
    prompt_specs = level
    train_paths, dataset_name = load_data(dataset,"SemSAM", prompt_specs, check_existing)
    out_path = f"{dataset_name}_SemSAM_{level}"

    if level == "high":
        level_list = [1,2,3]
    elif level == "low":
        level_list = [4,5,6]

    # create outdir if not exists
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    stats = {}

    # accumulate stats
    total_no_masks_count = 0
    total_img_wo_masks = []

    for img_path in tqdm.tqdm(train_paths):
        s, ipath, no_masks_count, img_wo_masks = generate_mask(img_path, out_path, create_stats, create_segments, dataset, level_list)
        stats[ipath] = s

        total_no_masks_count += no_masks_count
        total_img_wo_masks += img_wo_masks

    print("Mask generation completed.")
    print(f"Total stats collected: {len(stats)}")

    print("Total no masks count:", total_no_masks_count)
    stats["total_no_masks_count"] = total_no_masks_count
    stats["total_img_wo_masks"] = total_img_wo_masks

    # save stats
    if os.path.exists(f"stats/{dataset}_SemSAM_{level}_STATS.torch"):
        old_stats = torch.load(f"stats/{dataset}_SemSAM_{level}_STATS.torch")
        stats.update(old_stats)   
    torch.save(stats, f"stats/{dataset}_SemSAM_{level}_STATS.torch")
    print(len(stats))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to create segments using Semantic-SAM")
    parser.add_argument('--dataset', type=str, required=True, help='The dataset to use. Please select from CUB, ImageNet, ImageNette, ImageWoof')
    parser.add_argument('--level', type=str, required=True, help='The level of the model to use. Please select from "high", "low"')
    parser.add_argument('--create_segments', type=bool, required=False, default=True, help='Whether to create segment images or not')
    parser.add_argument('--create_stats', type=bool, required=False, default=True, help='Whether to create stats or not')
    parser.add_argument('--device', type=str, required=False, default="cuda:0", help='The device to use for detection')
    parser.add_argument('--check_existing', type=lambda x: (str(x).lower() == 'true'), required=False, default=True, help='Whether to check if the image has already been segmented')
    args = parser.parse_args()
    
    main(args.dataset, args.level, args.create_segments, args.create_stats, args.device, args.check_existing)