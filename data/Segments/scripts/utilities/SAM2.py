from utilities.segment_anything_2.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
import os
import numpy as np
from .segment import create_statistics, create_crops_masks, check_non_seg

def resize_image(image, target_size=1024):
    """Resize image while maintaining aspect ratio."""
    h, w, _ = image.shape
    if max(h, w) > target_size:
        scale = target_size / max(h, w)
        return cv2.resize(image, (int(w * scale), int(h * scale)))
    return image


def generate_mask(sam2, img_path, out_path, create_stats, create_segments, dataset_name):

    image = cv2.imread(img_path)
    ifolder = img_path.split("/")[-2]
    ipath = img_path.split("/")[-1].split(".")[0]
    # create folder if not exists
    if not os.path.exists(f"{out_path}/crops/{ifolder}"):
        os.makedirs(f"{out_path}/crops/{ifolder}")
    if not os.path.exists(f"{out_path}/masks/{ifolder}"):
        os.makedirs(f"{out_path}/masks/{ifolder}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_image(image, target_size=1024)
    # Generate the masks
    # Initialize the automatic mask generator with detailed parameters
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side=64,              # Increase to sample more points across the image # original: 32
        pred_iou_thresh=0.88,             # Increase for higher-quality masks # original 0.88
        stability_score_thresh=0.95,      # Higher stability threshold for more stable masks # ogirinal 0.9
        box_nms_thresh=0.5,              # Adjust Non Maximum Suppression to prevent overlapping masks # original 0.75
        min_mask_region_area=500         # Ignore small masks, set a lower value for more details # original 0

    )
    masks = mask_generator.generate(image)
    boxes = [m["bbox"] for m in masks]
    no_masks_count, img_wo_masks = check_non_seg(boxes, ipath)

    if create_segments:
        ipath = create_crops_masks(masks, image, out_path, ifolder, ipath)

    if create_stats: 
        
        s = create_statistics(boxes, img_path, dataset_name)
        
        return s, ipath, no_masks_count, img_wo_masks