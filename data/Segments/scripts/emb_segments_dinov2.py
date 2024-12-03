## conda env: clip_ex

# load packages
import torch
import torch.nn as nn
import tqdm
import glob
import os
from collections import Counter
from utilities.dinov2 import compute_embeddings, compute_embeddings_batched, make_classification_eval_transform
import argparse
from utilities.data_loading import load_segments

def load_emb_model(device):
    # load model
    model_name = "dinov2_vitl14" # alternatively: "facebook/dinov2-small", "facebook/dinov2-base","facebook/dinov2-giant"
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    preprocess = make_classification_eval_transform()
    model.eval() 
    model.to(device)
    return model, preprocess, model_name

def main(dataset, seg_model, prompt, device):
    # load embedding model
    model, preprocess, model_name = load_emb_model(device)
    # load segments to embed
    seg_paths = load_segments(dataset, seg_model, prompt)
    # iterate through the types of embeddings (crops, masks, annotated images)
    if dataset == "CUB":
        dataset = "CUB_200_2011"
    for i in seg_paths:
        seg_type = i.split("/")[-1]
        if seg_type in ["crops"]:#, "masks"]:
            if seg_model in ["GDINO", "SemSAM"]:
                out_file = f"Seg_embs/seg{seg_type[:-1]}_{dataset}_{seg_model}_{prompt}_{model_name}.torch"
            else:
                out_file = f"Seg_embs/seg{seg_type[:-1]}_{dataset}_{seg_model}_{model_name}.torch"
            print(out_file)
            # load paths of segments to embed
            segs = glob.glob(i+"/*/*.jpg")
            print(i+"/*/*.jpg")
            if not os.path.exists(out_file):
                print(f"computing embeddings for {seg_model} segments of type {seg_type}: {len(segs)} segments")
                compute_embeddings(segs, model, preprocess, out_file, device)
            else: 
                print("Embeddings already exist for {seg_type} segments of {dataset} using {seg_model} and {model_name}")
                print("Please check completeness")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute embeddings')
    parser.add_argument('--dataset', type=str, help='Dataset segments to embed')
    parser.add_argument('--seg_model', type=str, help='Model used to create segments. Please choose from: SAM, SAM2, SemSAM, GDINO')
    parser.add_argument('--prompts', default='None', type=str, help='Prompts used for segmentation (if applicable)')
    parser.add_argument('--device', default='cuda:1', type=str, help='Device to use for embedding')
    args = parser.parse_args()
    main(args.dataset, args.seg_model, args.prompts, args.device)

