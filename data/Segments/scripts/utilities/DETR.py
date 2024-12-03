from transformers import DetrImageProcessor
import os
from PIL import Image
import numpy as np
import torch
from .data_loading import load_CUB_bb

def create_statistics(boxes, img_path, dataset_name):
    sizes = [b[2]* b[3] for b in boxes]
    s = {"num_segments": len(boxes), "segments": boxes, "sizes": sizes, "overlap with gt": []}
    # Calculate if pred is overlapping with ground truth (boolean)
    if dataset_name == "CUB":
        bb_gt = load_CUB_bb()
        img_id = img_path.split("/")[-1].split(".")[0]
        gt = bb_gt[img_id]
        for b in boxes:
            overlap = False
            if b[0] <= gt[0] or b[2] >= gt[2] and b[1] <= gt[1] or b[3] >= gt[3]:
                overlap = True
            s["overlap with gt"].append(overlap)
    return s 

# Load the image to be segmented
def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")
    return image

def check_non_seg(boxes, ipath):
    no_masks_count = 0
    img_wo_masks = []
    if len(boxes) == 0:
        no_masks_count += 1
        img_wo_masks.append(ipath)
    return no_masks_count, img_wo_masks

def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    source: https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def generate_mask(img_path, out_path, create_stats, create_segments, model, device, dataset_name):
    ## prep folder for results
    ifolder = img_path.split("/")[-2]
    ipath = img_path.split("/")[-1].split(".")[0]
    # create folder if not exists
    if not os.path.exists(f"{out_path}/crops/"):
        os.makedirs(f"{out_path}/crops/")
    if not os.path.exists(f"{out_path}/masks/"):
        os.makedirs(f"{out_path}/masks/")
    if not os.path.exists(f"{out_path}/crops/{ifolder}"):
        os.makedirs(f"{out_path}/crops/{ifolder}")
    if not os.path.exists(f"{out_path}/masks/{ifolder}"):
        os.makedirs(f"{out_path}/masks/{ifolder}")
    
    # load preprocessing for DETR
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic")
    # load and prep image
    original_image = load_image(img_path)
    inputs = processor(images=original_image, return_tensors="pt")

    # compute the segmentation mask
    with torch.no_grad():
        outputs = model(**inputs)

    # post-process the results
    # Extract segmentation masks and bounding boxes
    masks = processor.post_process_segmentation(outputs, target_sizes=[original_image.size[::-1]])[0]
    masks = masks["masks"].cpu()
    boxes = masks_to_boxes(masks)
    

    # Loop through the masks and check stats
    no_masks_count, img_wo_masks = check_non_seg(masks, ipath)

    # create cropped / masked images
    if create_segments:
        counter = 0
        for i, (box, mask) in enumerate(zip(boxes, masks)):
            try:
                # Extract the bounding box and mask
                x1, y1, x2, y2 = map(int, box)  # Get bounding box coordinate

                # Create crop of the original image based on bounding box
                cropped = original_image.crop((x1, y1, x2, y2))
                cropped.save(f"{out_path}/crops/{ifolder}/{ipath}_OUT_CROP_{counter}_X_.jpg")
                    

                # Create an image from the mask
                mask = masks[i]
                masked_org = Image.fromarray(np.where(mask[...,None], original_image, 0))
                masked_crop = masked_org.crop((x1, y1, x2, y2))
                masked_crop.save(f"{out_path}/masks/{ifolder}/{ipath}_OUT_MASK_{counter}_X_.jpg")
                counter += 1
            except:
                print(ipath, box)

    if create_stats: 
        s = create_statistics(boxes, ipath, dataset_name)
        
        return s, ipath, no_masks_count, img_wo_masks
