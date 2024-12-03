## conda env: clip_ex

# load packages
import torch
import clip
import glob
import os
import argparse

from utilities.CLIP import fix_embeddings, compute_embeddings_batched
from utilities.data_loading import load_segments

def load_emb_model(emb, device):
    if emb == "CLIP-RN50":
        model_name = "RN50"
    elif emb == "CLIP-ViT-B16":
        model_name = "ViT-B/16"
    elif emb == "CLIP-ViT-L14":
        model_name = "ViT-L/14"
    model, preprocess = clip.load(model_name, device = device) # quickgelu is used by openai - we are using the official openai implementation
    model_name = f"CLIP-{model_name}".replace("/","")
    model.eval() 

    return model, preprocess, model_name

def main(emb, dataset, seg_model, prompt, device):
    # load embedding model
    model, preprocess, model_name = load_emb_model(emb, device)
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
                compute_embeddings_batched(segs, model, preprocess, out_file, device, batch_size=64)
            else: 
                embs = torch.load(out_file, weights_only= False, map_location="cpu")
                if len(embs) == len(segs):
                    #torch.save(embs, out_file)
                    print(f"Embeddings already exist for {seg_type} segments of {dataset} using {seg_model} and {model_name}")
                else:
                    print(f"Fixing embeddings for {seg_model} segments of type {seg_type}.")
                    seg_dict = {i.split("/")[-1]:i for i in segs}
                    missing_keys = [i for i in seg_dict.keys() if i not in embs.keys()]
                    print(len(missing_keys))
                    seg2emb = {i:seg_dict[i] for i in missing_keys}
                    missing_embs = fix_embeddings(seg2emb, model, preprocess, device)
                    embs.update(missing_embs)
                    torch.save(embs, out_file)
                    print(f"Embeddings updated for {seg_type} segments of {dataset} using {seg_model} and {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute embeddings')
    parser.add_argument('--emb', type=str, help='CLIP model to use: please choose from CLIP-RN50, CLIP-ViT-B16, CLIP-ViT-L14')
    parser.add_argument('--dataset', type=str, help='Dataset segments to embed')
    parser.add_argument('--seg_model', type=str, help='Model used to create segments. Please choose from: SAM, SAM2, SemSAM, GDINO')
    parser.add_argument('--prompts', default='None', type=str, help='Prompts used for segmentation (if applicable)')
    parser.add_argument('--device', default='cuda:1', type=str, help='Device to use for embedding')
    args = parser.parse_args()
    main(args.emb, args.dataset, args.seg_model, args.prompts, args.device)

