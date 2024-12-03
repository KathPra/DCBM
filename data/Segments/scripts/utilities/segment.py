from .data_loading import load_CUB_bb
import torch
import cv2
import numpy as np

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

def create_crops(boxes, phrases, image_source, out_path, ifolder, ipath):

    # Loop through the masks and display/save them
    for i, box in enumerate(boxes):
        try:
            # Extract the bounding box and mask
            h, w, _ = image_source.shape
            box = box * torch.Tensor([w, h, w, h])
            x, y, w, h = box.cpu().numpy().astype(int)
            # create crop of images
            cropped = image_source[y:y+h, x:x+w]
            file_name = f"{out_path}/crops/{ifolder}/{ipath}_OUT_CROP_{i}_{phrases[i]}.jpg"
            cv2.imwrite(file_name, cropped)
        except:
            print(ipath, box)
            # print error in try except block to avoid stopping the script

    return ipath

def check_non_seg(boxes, ipath):
    no_masks_count = 0
    img_wo_masks = []
    if len(boxes) == 0:
        no_masks_count += 1
        img_wo_masks.append(ipath)
    return no_masks_count, img_wo_masks


def create_crops_masks(masks, image, out_path, ifolder, ipath):
    # Loop through the masks and display/save them
    for i, mask in enumerate(masks):
        try:
            # Extract the bounding box and mask
            x, y, w, h = map(int, mask["bbox"])
            # create crop of images
            cropped = image[y:y+h, x:x+w]
            cv2.imwrite(f"{out_path}/crops/{ifolder}/{ipath}_OUT_CROP_{i}_X_.jpg", cropped)
            # create crop of mask
            seg_mask = mask["segmentation"][y:y+h, x:x+w]
            seg_mask = seg_mask.astype(bool)
            # create an image based on the segmentation mask
            # Multiply the mask with the image
            masked_crop = np.where(seg_mask[...,None], cropped, 0)
            # Save the mask as an image file
            cv2.imwrite(f"{out_path}/masks/{ifolder}/{ipath}_OUT_MASK_{i}_X_.jpg", masked_crop)
        except:
            print(ipath, mask["bbox"])

    return ipath
