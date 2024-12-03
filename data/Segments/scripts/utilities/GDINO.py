from utilities.GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate 
import os
import cv2
from .segment import create_statistics, create_crops, check_non_seg


def generate_mask(img_path, out_path, create_stats, create_segments, model, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD, device, dataset_name):

    image_source, image = load_image(img_path)
    ifolder = img_path.split("/")[-2]
    ipath = img_path.split("/")[-1].split(".")[0]
    # create folder if not exists
    if not os.path.exists(f"{out_path}/crops/"):
        os.makedirs(f"{out_path}/crops/")
    if not os.path.exists(f"{out_path}/annotated_imgs/"):
        os.makedirs(f"{out_path}/annotated_imgs/")
    if not os.path.exists(f"{out_path}/crops/{ifolder}"):
        os.makedirs(f"{out_path}/crops/{ifolder}")
    # Generate the masks
    boxes, logits, phrases = predict( model=model, image=image, caption=TEXT_PROMPT, box_threshold=BOX_TRESHOLD, text_threshold=TEXT_TRESHOLD, device=device ) 
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases) 
    cv2.imwrite(f"{out_path}/annotated_imgs/{ipath}_OUT.jpg", annotated_frame)

    no_masks_count, img_wo_masks = check_non_seg(boxes, ipath)

    if create_segments:
       ipath = create_crops(boxes, phrases, image_source, out_path, ifolder, ipath)

    if create_stats: 
        s = create_statistics(boxes, ipath, dataset_name)
        
        return s, ipath, no_masks_count, img_wo_masks

def prep_prompt(prompt_name):
    # load part names
    with open(f"prompts/{prompt_name}.txt") as f:
        parts = f.readlines()
    parts = [i.strip() for i in parts]
    # prepare for grounding dino
    TEXT_PROMPT = " . ".join(parts)
    TEXT_PROMPT = TEXT_PROMPT+" ."
    print(TEXT_PROMPT)
    return TEXT_PROMPT