# DCBM: Data-Efficient Visual Concept Bottleneck Models
You can find our paper on **[arXiv](https://arxiv.org/abs/2412.11576)** 📄.

**Accepted at ICML 2025** 🎉

Authors: Katharina Prasse*, Patrick Knab*, Sascha Marton, Christian Bartelt, and Margret Keuper 

\* Equal Contribution

![GCBM-Pipeline](data/Exemplary_explanations/pipeline.png)

---

## Overview

This repository is organized into two main components:

1. **Concept Extraction**
2. **DCBM Training and Visualization**

---

### 1. Concept Extraction

#### Step 1: Select Segmentation Models

Choose segmentation models based on your computational budget and the latest advancements. GDINO is the fastest model in our evaluation. We provide `.yml` files for all segmentation methods used in this work.

Build **SAM2**, **GroundingDINO**, and **Semantic SAM** from source, as suggested in their respective GitHub repositories. Place the repositories in the `utilities` folder and adjust the paths as necessary.

#### Step 2: Run the Segmentation Method

Execute the segmentation method by running the respective script in the `Segments` folder:

```bash
python scripts/create_segments_SAM2.py --dataset cifar100 --device cuda:0
```

#### Step 3: Embed Segments

Embed the segments into the selected embedding space by running the corresponding script in the `Segments` folder. We primarily report on CLIP models [RN-50, ViT-B16, ViT-L14] and save embeddings to reuse in DCBM training for efficiency.

```bash
python scripts/emb_segments_CLIP.py --dataset cifar100 --device cuda:0 --emb CLIP-RN50 --seg_model SAM2
```

---

### 2. DCBM Training and Visualization

#### Step 1: Create a Validation Set

For datasets without a provided validation set, create one by splitting the training set. Following common practices, we evaluate on the ImageNet-Val split, so we need a new set for hyperparameter validation.

Split the training set into training and validation sets (90:10). Ensure the test set is named `test`, as some datasets have `train` and `val` sets.

Adjust the dataset path in the script to your existing training set and the new validation set.

```bash
python data/Datasets/scripts/create_val.py
```

#### Step 2: Embed Images

Embed images to reuse in DCBM training for time and resource efficiency.

```bash
python data/Datasets/scripts/embed_CLIP.py
```

#### Step 3: Train DCBM

With all necessary files correctly stored in the `data` directory, initiate DCBM training:

```bash
python experiments/dcbm_testing.py
```

**Necessary Files in `data/`:**

1. `classes/`  
   Contains class information for the dataset (required for new datasets).
2. `Datasets/`  
   Contains the training, testing, and validation datasets.
3. `Embeddings/`  
   - Subset of embeddings (if not segmenting all training images).
   - Embedded files for test, validation, and training sets.
4. `Segments/`  
   - Contains segmented images.
   - Includes embeddings in the `Seg_embs/` subdirectory.
   - Includes detection/segmentation statistics in the `stats/` subdirectory.

We also provide multiple Jupyter notebooks in `dcbm_training/` to facilitate training and hyperparameter tuning.

#### Using Your Own Datasets

To apply the code to a new dataset, modify the relevant sections in `utils/dcbm.py` as indicated by the `TODO` comments.

#### DCBM Visualization

Explore interactive explanations with DCBM across different datasets using the Jupyter notebooks in `interpretation/`.

---

## Installation

We recommend create a conda environment with `Python=3.11`. 
Install the required dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

This environment supports the creation of CLIP embeddings. We have also created specialized environments for generating DINOv2 embeddings and using various segmentation models.

---

## Folder Structure

The repository is structured as follows:

```bash
.
├── concept_extraction/
├── dcbm_training/
├── data/
│   ├── classes/
│   ├── Datasets/
│   ├── Embeddings/
│   ├── Segments/
├── experiments/
├── utils/
├── interpretation/
├── requirements.txt
└── README.md
```


- `concept_extraction/`: Scripts and modules for extracting concepts.
- `dcbm_training/`: Code for training the DCBM model.
- `data/`: Directories for classes, concepts, datasets, embeddings, and segments.
- `experiments/`: Code for experiments detailed in the main paper and supplementary material.
- `utils/`: Helper scripts for running experiments.
- `interpretation/`: Tools for visualizing DCBM explanations.

**Important:** Some folders in the `data/` directory may be empty. Due to size limitations, we couldn't upload the complete dataset and corresponding concepts. We uploaded the empty structure to ensure all scripts run smoothly.

To run the scripts correctly, please add the required datasets before execution.

---

## Reproducibility

| Dataset | # Clusters *k* | Batch size | 
|---|---|---|
| ImageNet | 4096 | 512 |
| Places365 | 2048 | 512 | 
| CUB-200-2011 | 2048 | 32 | 
| CIFAR-10 | 256  | 32 | 
| CIFAR-100 | 2048 | 32 | 
| MiT-States | 2048 | 32 | 
| ClimateTV (animals) | 2048 | 32 |
| AWA2 | 2048 | 32 |
| CelebA | 2048 | 32 |


We train all DCBMs with learning rate: 1e-4, sparsity parameter λ: 1e-4, clustering algorithm: K-Means, centroid type: median of cluster embeddings, concept proposal images: 50 images/class (all images used if a class has <50, e.g. CUB), epochs 200

--- 

For more details, please refer to the sections and scripts within the repository or consult the paper.

--- 

## Citation

If you use our code, please cite us

```tex
@inproceedings{
dcbm2025,
title={{DCBM}: Data-Efficient Visual Concept Bottleneck Models},
author={Prasse, Katharina and Knab, Patrick and Marton, Sascha and Bartelt, Christian and Keuper, Margret},
booktitle={International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=BdO4R6XxUH}
}
```
