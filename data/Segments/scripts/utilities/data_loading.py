import glob
import torch

def extract_subset(all_train_paths, dataset_name):
    # open subset file
    with open(f"../Embeddings/subsets/{dataset_name}/{dataset_name}_rand_50.txt") as f:
        train_subset = f.readlines()
    # create dict from all training images
    train_dict = {i.split("/")[-1]: i for i in all_train_paths}
    all_train_paths = None

    # create training subset based on random images
    train_paths = [train_dict[i.strip()] for i in train_subset]
    train_dict = None
    train_subset = None
    return train_paths


def load_CUB_bb():
    # load bounding boxes for CUB (based on numeric image id)
    with open("../Datasets/CUB_200_2011/bounding_boxes.txt") as f:
        bb_gt = f.readlines() # id path
    # create dict mapping image id (numeric) to bounding box
    with open("../Datasets/CUB_200_2011/images.txt") as f:
        images = f.readlines() # id path

    # create dict mapping image id to image file name
    img_dict = {i.split(" ")[0]: i.split(" ")[1].strip() for i in images}

    bb_gt = {img_dict[i.split(" ")[0]].split("/")[-1].split(".")[0]: [int(j.split(".")[0]) for j in i.split(" ")[1:]] for i in bb_gt}
    return bb_gt


def load_data(dataset_name, model, prompt_specs, check_existing):
    if dataset_name == "CUB":
        dataset_name = "CUB_200_2011"
        # load file containing all training images
        with open("../Datasets/CUB_200_2011/train_imgs.txt") as f:
            train_imgs = f.readlines()
        # general image path
        img_dir = "../Datasets/CUB_200_2011/images/"
        # create list of image paths for training
        train_paths = [img_dir+i.strip() for i in train_imgs]

    if dataset_name == "ImageNette":
        # create dict with key: img id and value: path
        all_train_paths = glob.glob("../Datasets/imagenette2/train/*/*.JPEG")
        train_paths = extract_subset(all_train_paths, dataset_name)

    if dataset_name == "ImageWoof":
        all_train_paths = glob.glob("../Datasets/imagewoof2/train/*/*.JPEG")
        train_paths = extract_subset(all_train_paths, dataset_name)

    if dataset_name == "ImageNet":
        all_train_paths = glob.glob("../Datasets/ImageNet/ILSVRC/Data/CLS-LOC/train/*/*.JPEG")
        train_paths = extract_subset(all_train_paths, dataset_name)

    if dataset_name == "Places365":
        all_train_paths = glob.glob("../Datasets/Places365/Data/train/*/*.jpg")
        train_paths = extract_subset(all_train_paths, dataset_name)

    if dataset_name == "cifar10":
        all_train_paths = glob.glob("../Datasets/cifar10/train/*/*.png")
        train_paths = extract_subset(all_train_paths, dataset_name)
    
    if dataset_name == "cifar100":
        all_train_paths = glob.glob("../Datasets/cifar100/train/*/*.png")
        train_paths = extract_subset(all_train_paths, dataset_name)
    
    if dataset_name == "ClimateTV":
        train_paths = glob.glob("../Datasets/ClimateTV/train/*/*.jpg")

    if dataset_name == "MiT-States":
        with open("../Datasets/MiT-States/train_imgs.txt") as f:
            train_imgs = f.readlines()
        img_dir = "../Datasets/MiT-States/release_dataset/images/"
        train_paths = [img_dir+i.strip() for i in train_imgs]

    if check_existing:
        # check if some images have already been processed
        if prompt_specs is None:
            ## successfully processed images
            complete = glob.glob(f"{dataset_name}_{model}/*/*/*")
            ## list of images without segments
            try:
                stats = torch.load(f"Seg_embs/{dataset_name}_{model}_STATS.torch")
            except:
                stats = {"total_img_wo_masks": []}
        else: 
            complete = glob.glob(f"{dataset_name}_{model}_{prompt_specs}/*/*/*")
            ## list of images without segments
            try:
                stats = torch.load(f"Seg_embs/{dataset_name}_{model}_{prompt_specs}_STATS.torch")
            except:
                stats = {"total_img_wo_masks": []}

        complete = [i.split("/")[-1] for i in complete]
        complete = [i.split("_OUT_")[0] for i in complete]

        no_seg_list = stats["total_img_wo_masks"]
        complete = complete + no_seg_list

        # keep only training images that are not yet processed
        train_paths_ver = []
        for path in train_paths:
            name = path.split("/")[-1].split(".")[0]
            if name not in complete:
                train_paths_ver.append(path)
    else: train_paths_ver = train_paths

    print("Creating segments for ", len(train_paths_ver), "images")
    if len(train_paths_ver) == 0:
        print("No images to segment")
        exit()
    
    return train_paths_ver, dataset_name

def load_segments(dataset_name, model, prompt_specs):
    if dataset_name == "CUB":
        dataset_name = "CUB_200_2011"

    if prompt_specs == "None":
        seg_paths = glob.glob(f"{dataset_name}_{model}/*")
    elif prompt_specs is not None: 
        seg_paths = glob.glob(f"{dataset_name}_{model}_{prompt_specs}/*")
    print(seg_paths)
    return seg_paths