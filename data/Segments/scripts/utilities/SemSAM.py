import cv2
import os
import numpy as np
from utilities.Semantic_SAM.semantic_sam import prepare_image, plot_results, build_semantic_sam, SemanticSamAutomaticMaskGenerator
from .segment import create_statistics, create_crops_masks, check_non_seg


def generate_mask(img_path, out_path, create_stats, create_segments, dataset_name, level):

    original_image, image = prepare_image(img_path)
    ifolder = img_path.split("/")[-2]
    ipath = img_path.split("/")[-1].split(".")[0]
    # create folder if not exists
    if not os.path.exists(f"{out_path}/crops/{ifolder}"):
        os.makedirs(f"{out_path}/crops/{ifolder}")
    if not os.path.exists(f"{out_path}/masks/{ifolder}"):
        os.makedirs(f"{out_path}/masks/{ifolder}")
    if not os.path.exists(f"{out_path}/annotated_imgs/"):
        os.makedirs(f"{out_path}/annotated_imgs/")

    # Generate the masks
    # Initialize the automatic mask generator with detailed parameters
    mask_generator = SemanticSamAutomaticMaskGenerator(
        build_semantic_sam(
            model_type='L', 
            ckpt='scripts/weights/swinl_only_sam_many2many.pth'),
            level=level
            
    ) # model_type: 'L' / 'T', depends on your checkpint

    masks = mask_generator.generate(image)

    plot_results(masks, original_image, f"{out_path}/annotated_imgs/{ipath}_OUT.jpg")
    boxes = [m["bbox"] for m in masks]

    no_masks_count, img_wo_masks = check_non_seg(boxes, ipath)

    if create_segments:
        ipath = create_crops_masks(masks, original_image, out_path, ifolder, ipath)

    if create_stats: 
        s = create_statistics(boxes, img_path, dataset_name)
        
        return s, ipath, no_masks_count, img_wo_masks