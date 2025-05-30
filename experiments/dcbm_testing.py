import os
import argparse
import wandb
import re
import clip

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# add the parent directory to the path
import sys
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)
                
from utils.dcbm import *


# ----------------- Constants -----------------
embed_path = "../data/Embeddings/"
dataset = "cifar100"
class_labels_path = "../data/classes/cifar100_classes.txt"
segment_path = "../data/Segments/"
selected_image_concepts = "../data/Embeddings/subsets"

# ----------------- Hyperparameters -----------------
model_name = "CLIP-ViT-L14"  # "CLIP-ViT-L14", "CLIP-RN50"

cluster_method = "kmeans"     # "hierarchical", "kmeans"
centroid_method = "median"    # "mean", "median"
concept_per_class = 50      # How many images for each class: 5,10,20,50, None

one_hot = False
epochs = 200
batch_size = 32
crop = False                  # True without background

use_wandb = False
project = "YOUR_PROJECT_NAME"        # Define your own project name within wandb
device = "cpu"

def run_training(cbm):
    """Preprocess data and train the CBM model with different hyperparameters."""
    cbm.preprocess_data(type_="standard", label_type=one_hot)
    for lambda_1, lr in [(1e-4, 1e-4), (1e-3, 1e-3), (1e-3, 1e-4)]: #define here your own hyperparameters
        cbm.train(
            num_epochs=epochs,
            lambda_1=lambda_1,
            lr=lr,
            device=device,
            batch_size=batch_size,
            project=project,
            to_print=False,
            early_stopping_patience=None,
            one_hot=one_hot,
            use_wandb=use_wandb,
        )

def run_experiment(segmentation_technique, concept_name, clusters_list, load_concepts_first):
    """Run experiments with specified segmentation techniques and clustering options."""
    # Initial training without loading concepts and clustering
    if not load_concepts_first:
        cbm = CBM(embed_path, dataset, model_name, class_labels_path, device=device)
        run_training(cbm)
    # Training with loaded concepts and clustering
    for clusters in clusters_list:
        cbm = CBM(embed_path, dataset, model_name, class_labels_path, device=device)
        cbm.load_concepts(
            segment_path,
            segmentation_technique,
            concept_name,
            selected_image_concepts,
            concept_per_class,
            crop=crop,
        )
        if clusters is not None:
            cbm.cluster_image_concepts(cluster_method, clusters)
        else:
            cbm.clustered_concepts = cbm.image_segments
        cbm.centroid_concepts(centroid_method)
        run_training(cbm)

# ----------------- Experiments -----------------
experiments = [
    {
        'segmentation_technique': 'SAM2',
        'concept_name': None,
        'clusters_list': [128, 256, 512, 1024, 2048], #define here your own cluster sizes
        'load_concepts_first': True
    },
    
    # ---- Attention! ----
    # you need to have the segments for GDINO, MaskRCNN or any other segmentation technique to run those!
    
    {
        'segmentation_technique': 'GDINO',
        'concept_name': 'partimagenet', # used promt for GDINO
        'clusters_list': [128, 256],
        'load_concepts_first': True
    },
    {
        'segmentation_technique': 'MaskRCNN',
        'concept_name': None,
        'clusters_list': [128, 256],
        'load_concepts_first': True
    },
]

for exp in experiments:
    run_experiment(
        segmentation_technique=exp['segmentation_technique'],
        concept_name=exp['concept_name'],
        clusters_list=exp['clusters_list'],
        load_concepts_first=exp['load_concepts_first']
    )
