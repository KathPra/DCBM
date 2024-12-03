import os
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torch

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
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])  # Convert image to tensor & convert to rgb
    image_tensor = transform(image).unsqueeze(0) # Add batch dimension
    return image_tensor, image

def check_non_seg(boxes, ipath):
    no_masks_count = 0
    img_wo_masks = []
    if len(boxes) == 0:
        no_masks_count += 1
        img_wo_masks.append(ipath)
    return no_masks_count, img_wo_masks

# Calculate IoU between two boxes
def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

# Perform Non-Maximum Suppression
def non_maximum_suppression(boxes, scores, iou_threshold=0.5):
    sorted_indices = np.argsort(scores)[::-1]  # Sort by scores descending
    selected_indices = []

    while len(sorted_indices) > 0:
        # Select the box with the highest score
        current_index = sorted_indices[0]
        selected_indices.append(current_index)

        # Compare with the rest
        remaining_boxes = boxes[sorted_indices[1:]]
        current_box = boxes[current_index]

        # Remove boxes that have IoU greater than the threshold
        sorted_indices = np.array([
            idx for idx in sorted_indices[1:] 
            if calculate_iou(current_box, boxes[idx]) <= iou_threshold
        ])

    return selected_indices

def generate_mask(img_path, out_path, create_stats, create_segments, model, device, dataset_name):
    # derive image folder and image name
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
    
    # load and prep image
    inputs, original_image = load_image(img_path)
    inputs = inputs.to(device)
    model = model.to(device)

    # compute the segmentation mask
    with torch.no_grad():
        outputs = model(inputs)[0]

    # post-process the results
    boxes = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()  # Get confidence scores
    masks = outputs["masks"].cpu().numpy().squeeze(1)  # Get the masks

    # Apply NMS
    selected_indices = non_maximum_suppression(boxes, scores, iou_threshold=0.5)
    final_indices = []

    for num in selected_indices:
        b = boxes[num]
        if b[0] < b[2] or b[1] < b[3]:
            final_indices.append(num)  

    boxes = boxes[final_indices].tolist()  # Get the valid boxes
    masks = masks[final_indices]  # Get the valid masks
    
    # Loop through the masks and check stats
    no_masks_count, img_wo_masks = check_non_seg(boxes, ipath)

    # create cropped / masked images
    if create_segments:
        counter = 0
        for i, (box, mask) in enumerate(zip(boxes, masks)):
            try:
                # Extract the bounding box and mask
                x1, y1, x2, y2 = map(int, box)  # Get bounding box coordinates
                # Create crop of the original image based on bounding box
                cropped = original_image.crop((x1, y1, x2, y2))
                cropped.save(f"{out_path}/crops/{ifolder}/{ipath}_OUT_CROP_{counter}_X_.jpg")
                    
                # Threshold mask (Mask R-CNN outputs are probabilistic)
                mask_np = (mask > 0.5).astype(np.uint8) * 255
                masked_crop = np.where(mask_np[...,None], original_image, 0)
                # Create an image from the mask
                masked_crop = Image.fromarray(masked_crop[y1:y2, x1:x2])
                masked_crop.save(f"{out_path}/masks/{ifolder}/{ipath}_OUT_MASK_{counter}_X_.jpg")
                counter += 1
            except:
                print(ipath, box)

    if create_stats: 
        s = create_statistics(boxes, ipath, dataset_name)
        
        return s, ipath, no_masks_count, img_wo_masks
