## conda env: clip_ex

import random
import argparse
import glob
import os

# set seed
random.seed(42)

def load_dataset(dataset):
    if dataset == "ImageNet":
        train_path = "../Datasets/ImageNet/ILSVRC/Data/CLS-LOC/train/"
        num_classes = 1000

    elif dataset == "ImageNette":
        train_path = "../Datasets/imagenette2/train/"
        num_classes = 10
    
    elif dataset == "ImageWoof":
        train_path = "../Datasets/imagewoof2/train/"
        num_classes = 10

    elif dataset == "CUB":
        with open("../Datasets/CUB_200_2011/train_imgs.txt","r") as f:
            train_path = f.readlines()
        train_path = [i.strip() for i in train_path]
        num_classes = 200

    elif dataset == "cifar10":
        train_path = "../Datasets/cifar10/train/"
        num_classes = 10

    elif dataset == "cifar100":
        train_path = "../Datasets/cifar100/train/"
        num_classes = 100
    
    elif dataset == "ClimateTV":
        train_path = "../Datasets/ClimateTV/train/"
        num_classes = 11

    elif dataset == "Places365":
        train_path = "../Datasets/Places365/Data/train/"
        num_classes = 365

    elif dataset == "MiT-States":
        with open("../Datasets/MiT-States/train_imgs.txt","r") as f:
            train_path = f.readlines()
        train_path = [i.strip() for i in train_path]
        num_classes = 245


    if dataset in ["ImageNet", "ImageNette", "ImageWoof", "cifar10", "cifar100", "Places365", "ClimateTV"]:
        classes = glob.glob(train_path+"*")
        
    elif dataset in ["CUB"]:
        classes = set([i.split("/")[0] for i in train_path])

    elif dataset in ["MiT-States"]:
        classes = set([i.split("/")[1].split("_")[0] for i in train_path])
        
    if len(classes) != num_classes:
            raise ValueError(f"Number of classes is not {num_classes}, but {len(classes)}")

    return classes, train_path


def path2id(path_list):
    return [i.split("/")[-1] for i in path_list]


def save_subset(subset, path):
    if not os.path.exists(path):
        with open(path, "w") as f:
            for i in subset:
                f.write(i+"\n")
    else:
        path = path.replace(".txt", "_updated.txt")
        with open(path, "w") as f:
            for i in subset:
                f.write(i+"\n")

def list2classdict(class_paths):
    class_dict = {}
    for i in class_paths:
        class_id = i.split("/")[-2]
        if class_id not in class_dict:
            class_dict[class_id] = [i]
        else:
            class_dict[class_id].append(i)
    return class_dict

def list2classdictspecial(class_paths):
    class_dict = {}
    for i in class_paths:
        class_id = i.split("/")[-1].split("_")[0]
        if class_id not in class_dict:
            class_dict[class_id] = [i]
        else:
            class_dict[class_id].append(i)
    for i in class_dict:
        if len (class_dict[i]) < 5:
            print(i)
    return class_dict

def check_existing(dataset):
    path = "subsets/archive/"+dataset+"_rand_5.txt"
    if os.path.exists(path):
        return True
    else:
        return False
    
def load_existing(dataset):
    train_paths = []
    try:
        print("Loading existing subset of size 50")
        with open("subsets/archive/"+dataset+"_rand_50.txt") as f:
            paths = f.readlines()
            
    except:
        print("Loading existing subset of size 25")
        with open("subsets/archive/"+dataset+"_rand_25.txt") as f:
            paths = f.readlines()
    total_size = len(paths)
    paths = [i.strip() for i in paths]
    for i in paths:
        train_paths.append(i)
    return train_paths, total_size

def sample_from_existing(existing_sample, sample_size, prev_size):
    # sample from existing subset for each class
    # take batches of 5, 10, 25, 50 and sample from each batch
    total_length = len(existing_sample)
    rand = []
    for ndx in range(0, total_length, prev_size):
        current = existing_sample[ndx:min(ndx + prev_size, total_length)]
        rand.extend(random.sample(current, sample_size))
    return rand

 

# load data
def main(dataset):
    print("Creating subset for", dataset)
    sample_exists = check_existing(dataset)
    class_paths, train_paths = load_dataset(dataset)

    # if subset exists replace all paths with largest existing subset
    if sample_exists:
        train_paths, total_size = load_existing(dataset)
        max_size = total_size / len(class_paths)


    if sample_exists:
        sample_sizes = [5, 10, 25, 50]
        samples = [x for x in sample_sizes if x < max_size]
        if len(samples) == 3:
            print("creating subsets of sizes 25,10,5")
            rand25 = sample_from_existing(train_paths, 25, 50)
            rand10 = sample_from_existing(rand25, 10, 25)
            rand5 = sample_from_existing(rand10, 5, 10)
            save_subset(path2id(rand25), "subsets/"+dataset+"/"+dataset+"_rand_25_updated.txt")

        elif len(samples) == 2:
            print("creating subsets of sizes 10,5")
            rand10 = sample_from_existing(train_paths, 10, 25)
            rand5 = sample_from_existing(rand10, 5, 10)
        save_subset(path2id(rand5), "subsets/"+dataset+"/"+dataset+"_rand_5.txt")
        save_subset(path2id(rand10), "subsets/"+dataset+"/"+dataset+"_rand_10.txt")
    
    else:

        rand50 = []
        rand25 = []
        rand10 = []
        rand5 = []
    
        # sample images from each class
        if dataset in ["ImageNet", "ImageNette", "ImageWoof", "cifar10", "cifar100", "Places365", "ClimateTV"]:
            for i in class_paths:
                imgs = glob.glob(i+"/*")
                # create random subset of images
                if len(imgs) >= 50:
                    rand50_i = random.sample(imgs, 50)
                    rand50.extend(rand50_i)
                    rand25_i = random.sample(rand50_i, 25)
                    rand25.extend(rand25_i)
                else:
                    rand25_i = random.sample(imgs, 25)
                    rand25.extend(rand25_i)
                rand10_i = random.sample(rand25_i, 10)
                rand10.extend(rand10_i)
                rand5.extend(random.sample(rand10_i, 5))

        elif dataset in ["CUB"]:
            class_dict = list2classdict(train_paths)
            for imgs in class_dict.values():
                # create random subset of images
                if len(imgs) >= 50:
                    rand50_i = random.sample(imgs, 50)
                    rand50.extend(rand50_i)
                    rand25_i = random.sample(rand50_i, 25)
                    rand25.extend(rand25_i)
                else:
                    rand25_i = random.sample(imgs, 25)
                    rand25.extend(rand25_i)
                rand10_i = random.sample(rand25_i, 10)
                rand10.extend(rand10_i)
                rand5.extend(random.sample(rand10_i, 5))

        elif dataset in ["MiT-States"]:
            class_dict = list2classdictspecial(train_paths)
            for imgs in class_dict.values():
                # create random subset of images
                if len(imgs) >= 50:
                    rand50_i = random.sample(imgs, 50)
                    rand50.extend(rand50_i)
                    rand25_i = random.sample(rand50_i, 25)
                    rand25.extend(rand25_i)
                else:
                    rand25_i = random.sample(imgs, 25)
                    rand25.extend(rand25_i)
                rand10_i = random.sample(rand25_i, 10)
                rand10.extend(rand10_i)
                rand5.extend(random.sample(rand10_i, 5))


        # save subsets
        save_subset(path2id(rand5), "subsets/"+dataset+"/"+dataset+"_rand_5.txt")
        save_subset(path2id(rand10), "subsets/"+dataset+"/"+dataset+"_rand_10.txt")
        save_subset(path2id(rand25), "subsets/"+dataset+"/"+dataset+"_rand_25.txt")
        if rand50:
            save_subset(path2id(rand50), "subsets/"+dataset+"/"+dataset+"_rand_50.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to run different models")
    parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset of which a subset should be created')
    
    args = parser.parse_args()
    
    main(args.dataset)