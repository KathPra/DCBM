## conda env: models

## import packages
import os
import torch
import argparse
import tqdm

from transformers import DetrForSegmentation
from utilities.data_loading import load_data
from utilities.DETR import generate_mask

torch.cuda.empty_cache() 

def main(dataset,create_segments=True, create_stats=True, device="cuda:0", check_existing=True):

    # load data
    prompt_specs = None
    train_paths, dataset_name = load_data(dataset, 'DETR',prompt_specs, check_existing)
    out_path = f"{dataset_name}_DETR"

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # load GDINO model
    model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")

    # print confirmation of task
    print("create segment images is set to:", create_segments, "and create stats is set to:", create_stats)
    
    # create empty dict for detection statistics
    stats = {}

    # accumulate stats
    total_no_masks_count = 0
    total_img_wo_masks = []

    # loop through images
    for img_path in tqdm.tqdm(train_paths):
        s, ipath, no_masks_count, img_wo_masks = generate_mask(img_path, out_path, create_stats, create_segments, model, device, dataset)
        stats[ipath] = s

        total_no_masks_count += no_masks_count
        total_img_wo_masks += img_wo_masks

    print("Mask generation completed.")
    print(f"Total stats collected: {len(stats)}")
    
    print("Total no masks count:", total_no_masks_count)

    stats["total_no_masks_count"] = total_no_masks_count
    stats["total_img_wo_masks"] = total_img_wo_masks


    # save stats
    if os.path.exists(f"stats/{dataset}_DETR_STATS.torch"):
        old_stats = torch.load(f"stats/{dataset}_DETR_STATS.torch")
        stats.update(old_stats)   
    torch.save(stats, f"stats/{dataset}_DETR_STATS.torch")
    print(len(stats))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to create segments using DETR")
    parser.add_argument('--dataset', type=str, required=True, help='The dataset to use. Please select from CUB, ImageNet, ImageNette, ImageWoof')
    parser.add_argument('--create_segments', type=lambda x: (str(x).lower() == 'true'), required=False, default=True, help='Whether to create segment images or not')
    parser.add_argument('--create_stats', type=lambda x: (str(x).lower() == 'true'), required=False, default=True, help='Whether to create stats or not')
    parser.add_argument('--device', type=str, required=False, default="cuda:0", help='The device to use for detection')
    parser.add_argument('--check_existing', type=lambda x: (str(x).lower() == 'true'), required=False, default=True, help='Whether to check if the image has already been segmented')
    args = parser.parse_args()
    
    main(args.dataset, args.create_segments, args.create_stats, args.device, args.check_existing)