## conda env: grounding_dino

## import packages
import os
import torch
import argparse
import tqdm

from utilities.GroundingDINO.groundingdino.util.inference import load_model
from utilities.data_loading import load_data
from utilities.GDINO import generate_mask, prep_prompt

torch.cuda.empty_cache() 

def main(prompt,dataset,threshold="standard", create_segments=True, create_stats=True, device="cuda:0", check_existing=True):
    # set model threshold
    if threshold =="lowthresh":
        BOX_TRESHOLD = 0.25 # org: 0.35
        TEXT_TRESHOLD = 0.25 # org. 0.25
    elif threshold == "standard":
        BOX_TRESHOLD = 0.35
        TEXT_TRESHOLD = 0.25

    # set prompt specs
    if threshold == "lowthresh":
        prompt_specs = prompt+"-"+threshold
    else:
        prompt_specs = prompt
    # load data
    train_paths, dataset_name = load_data(dataset, 'GDINO', prompt_specs, check_existing)
    out_path = f"{dataset_name}_GDINO_{prompt_specs}"

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # load GDINO model
    model = load_model("scripts/utilities/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "scripts/utilities/GroundingDINO/weights/groundingdino_swint_ogc.pth") 

    # print confirmation of task
    print(f"Box threshold: {BOX_TRESHOLD} and Text threshold: {TEXT_TRESHOLD}")
    print("create segment images is set to:", create_segments, "and create stats is set to:", create_stats)
    
    # create empty dict for detection statistics
    stats = {}

    # create prompts for image segmentation
    TEXT_PROMPT = prep_prompt(prompt)

    # accumulate stats
    total_no_masks_count = 0
    total_img_wo_masks = []

    # loop through images
    for img_path in tqdm.tqdm(train_paths):
        s, ipath, no_masks_count, img_wo_masks = generate_mask(img_path, out_path, create_stats, create_segments, model, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD, device, dataset)
        stats[ipath] = s

        total_no_masks_count += no_masks_count
        total_img_wo_masks += img_wo_masks

    print("Mask generation completed.")
    print(f"Total stats collected: {len(stats)}")
    
    print("Total no masks count:", total_no_masks_count)

    stats["total_no_masks_count"] = total_no_masks_count
    stats["total_img_wo_masks"] = total_img_wo_masks


    # save stats
    if os.path.exists(f"stats/{dataset}_GDINO_{prompt_specs}_STATS.torch"):
        old_stats = torch.load(f"stats/{dataset}_GDINO_{prompt_specs}_STATS.torch")
        stats.update(old_stats)   
    torch.save(stats, f"stats/{dataset}_GDINO_{prompt_specs}_STATS.torch")
    print(len(stats))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to create segments using GDINO")
    parser.add_argument('--prompt', type=str, required=True, help='The name of the prompt to use. Please select from sun, awa, CUB-parts, partimagenet, pascal-parts')
    parser.add_argument('--dataset', type=str, required=True, help='The dataset to use. Please select from CUB, ImageNet, ImageNette, ImageWoof')
    parser.add_argument('--threshold', type=str, required=False, default="standard", help='The threshhold the model uses for detection. Please select from standard, lowthresh')
    parser.add_argument('--create_segments', type=lambda x: (str(x).lower() == 'true'), required=False, default=True, help='Whether to create segment images or not')
    parser.add_argument('--create_stats', type=lambda x: (str(x).lower() == 'true'), required=False, default=True, help='Whether to create stats or not')
    parser.add_argument('--device', type=str, required=False, default="cuda:0", help='The device to use for detection')
    parser.add_argument('--check_existing', type=lambda x: (str(x).lower() == 'true'), required=False, default=True, help='Whether to check if the image has already been segmented')
    args = parser.parse_args()
    
    main(args.prompt, args.dataset, args.threshold, args.create_segments, args.create_stats, args.device, args.check_existing)