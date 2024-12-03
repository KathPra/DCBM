## conda env: clip_ex

import torch
import tqdm
import glob

# load ImageNet-R classes
inr_classes = glob.glob("../Datasets/ImageNet-R/imagenet-r/*")
inr_classes = set([i.split("/")[-1] for i in inr_classes])
print(len(inr_classes))

# load ImageNet Val embeddings (RN50)
emb = torch.load("images_ImageNet_test_CLIP-RN50.torch", map_location = "cpu")
print(len(emb))
print(emb[list(emb.keys())[0]])
emb_200 = {k: emb[k] for k,v in emb.items() if v[1] in inr_classes}
print(len(emb_200))
torch.save(emb_200, "images_ImageNet-200_test_CLIP-RN50.torch")
emb = None

# load ImageNet Val embeddings (ViT-B/16)
emb = torch.load("images_ImageNet_test_CLIP-ViT-B16.torch", map_location = "cpu")
print(len(emb))
print(emb[list(emb.keys())[0]][1])
emb_200 = {k: v for k,v in emb.items() if v[1] in inr_classes}
print(len(emb_200))
torch.save(emb_200, "images_ImageNet-200_test_CLIP-ViT-B16.torch")
emb = None

# load ImageNet Val embeddings (ViT-L/14)
emb = torch.load("images_ImageNet_test_CLIP-ViT-L14.torch", map_location = "cpu")
print(len(emb))
emb_200 = {k: emb[k] for k,v in emb.items() if v[1] in inr_classes}
print(len(emb_200))
torch.save(emb_200, "images_ImageNet-200_test_CLIP-ViT-L14.torch")
emb = None

# load ImageNet test embeddings (dinov2)
emb = torch.load("images_ImageNet_test_dinov2_vitl14.torch", map_location = "cpu")
print(len(emb))
emb_200 = {k: emb[k] for k,v in emb.items() if v[1] in inr_classes}
print(len(emb_200))
torch.save(emb_200, "images_ImageNet-200_test_dinov2_vitl14.torch")
emb = None