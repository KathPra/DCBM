import numpy as np
from scipy.special import softmax
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import collections
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import os
import matplotlib.pyplot as plt
import re
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from collections import Counter
import random
import glob


from tqdm import tqdm

import clip

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime 

import wandb
torch.manual_seed(5)
np.random.seed(5)

class CBM ():
    
    def __init__(self, 
                 embeddings_path, 
                 dataset,
                 model,
                 class_labels_path,
                 val_split = 0.1,
                 standard_scaler = True,
                 device = 'cpu',
                 to_save_concepts = False):
        
        self.device = device
        self.dataset = dataset
        self.model = None
        self.clip_model = model
        self.class_labels = None
    
        self.embeddings_path = embeddings_path
        self.embeddings_path_train = None
        self.embeddings_path_test = None
        self.embeddings_path_val = None
        self.embeddings_path_test_second = None
        self.segmentation_technique = None
        
        self.data_train_raw = None
        self.data_test_raw = None
        self.data_val_raw = None
        self.data_test_second_raw = None
        
        self.data_train_labels_raw = None
        self.data_test_labels_raw = None
        self.data_val_labels_raw = None
        self.data_test_labels_raw_second = None
        
        self.train_labels_int = None
        self.test_labels_int = None
        self.val_labels_int = None
        
        self.train_labels_one_hot = None
        self.test_labels_one_hot = None
        self.val_labels_one_hot = None
        
        self.data_train = None
        self.data_test = None
        self.data_val = None
        self.data_test_second = None
        
        self.class_num = "All"
        
        self.val_split = val_split
        
        self.standard_scaler = standard_scaler
        self.criterion = None
        
        self.image_segments = None
        self.image_segments_names = None
        self.clustered_concepts = None
        self.clustered_concepts_all = None
        self.cluster_technique = None
        self.centroid_method = None
        self.concept_name = None
        self.background = False
        
        self.scaler = None
        
        self.cv_scores = None
        
        self.to_save_concepts = to_save_concepts
        
        #initialize the embeddings
        self.construct_paths()
        self.load_data()
        self.extract_data_labels()
        self.load_labels_list(class_labels_path)
        
    
    def load_labels_list(self, path):
        #load txt file with labels
        class_labels_dict = {}
        class_dict = {}
        
        if path is not None:
            with open(path, 'r') as f:
                for i, line in enumerate(f):
                    class_labels_dict[i] = line.strip()
    
        # ---- TODO -----
        # Define own dataset
    
        if self.dataset == 'cub':
            for i, class_name in enumerate(class_labels_dict.keys()):
                class_dict[class_name] = class_labels_dict[i].split(".")[1]   
            class_dict = [class_dict[im_class] for im_class in class_dict.keys()]
        if self.dataset == 'cifar100':
            class_dict = [class_labels_dict[im_class] for im_class in class_labels_dict.keys()]
        if self.dataset == 'cifar10':
            class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
            class_dict = [class_dict[im_class] for im_class in class_dict.keys()]
        if self.dataset == "places365":
            class_dict = [class_labels_dict[im_class] for im_class in class_labels_dict.keys()]
        if self.dataset == "imagenette":
            class_dict = {0: 'tench', 1: 'English_springer', 2: 'cassette_player', 3: 'chain_saw', 4: 'church', 5: 'French_horn', 6: 'garbage_truck', 7: 'gas_pump', 8: 'golf_ball', 9: 'parachute'}
            class_dict = [class_dict[im_class] for im_class in class_dict.keys()]
        if self.dataset == "imagewoof":
            class_dict = {
                0: 'Shih-Tzu',
                1: 'Rhodesian_ridgeback',
                2: 'Beagle',
                3: 'English_foxhound',
                4: 'Border_terrier',
                5: 'Australian_terrier',
                6: 'Golden_retriever',
                7: 'Old_English_sheepdog',
                8: 'Samoyed',
                9: 'Dingo'
            }
            class_dict = [class_dict[im_class] for im_class in class_dict.keys()]
        if self.dataset == "climateTV":
            class_dict = {
                0: 'Amphibian_or_Reptile',
                1: 'Birds',
                2: 'Farm_animals',
                3: 'Fish',
                4: 'Insects',
                5: 'Land_mammal',
                6: 'Mixed_animals',
                7: 'No_animals',
                8: 'Pets',
                9: 'Polar_bear',
                10: 'Sea_mammal',
            }
            class_dict = [class_dict[im_class] for im_class in class_dict.keys()]
        if self.dataset == "imagenet":
            class_dict = [class_labels_dict[im_class] for im_class in class_labels_dict.keys()]
            
        if self.dataset == "mit_states":
            x = glob.glob("") # TODO add path to mit_states classes
            classes = set([i.split("/adj_")[-1] for i in x])
            class_dict = list(classes)
        
        self.class_labels = class_dict
            
    def construct_paths(self):
        
        #print("Constructing paths...")
        
        model_name = self.clip_model
        embeddings_parent_directory = self.embeddings_path
        dataset = self.dataset
        
        train_path = None
        test_path = None
        val_path = None
        test_path_second = None
        
        if dataset == "places365":
            local_name = f'images_Places365_train_{model_name}.torch' 
            train_path = os.path.join(embeddings_parent_directory, local_name)
            local_name = f'images_Places365_test_{model_name}.torch' 
            test_path = os.path.join(embeddings_parent_directory, local_name)
            local_name = f'images_Places365_val_{model_name}.torch' 
            val_path = os.path.join(embeddings_parent_directory, local_name)
        
        elif dataset == "imagenet":
            local_name = f'images_ImageNet_train_{model_name}.torch' 
            train_path = os.path.join(embeddings_parent_directory, local_name)    
            local_name = f'images_ImageNet_test_{model_name}.torch'
            test_path = os.path.join(embeddings_parent_directory, local_name)
            local_name = f'images_ImageNet_val_{model_name}.torch'
            val_path = os.path.join(embeddings_parent_directory, local_name)
            local_name = f'images_ImageNet-R_val_{model_name}.torch'
            test_path_second = os.path.join(embeddings_parent_directory, local_name)
            
        elif dataset == "cifar10":
            local_name = f'images_cifar10_train_{model_name}.torch' 
            train_path = os.path.join(embeddings_parent_directory, local_name)    
            local_name = f'images_cifar10_test_{model_name}.torch'
            test_path = os.path.join(embeddings_parent_directory, local_name)
            local_name = f'images_cifar10_val_{model_name}.torch'
            val_path = os.path.join(embeddings_parent_directory, local_name)
            
        elif dataset == "cifar100":
            local_name = f'images_cifar100_train_{model_name}.torch' 
            train_path = os.path.join(embeddings_parent_directory, local_name)    
            local_name = f'images_cifar100_test_{model_name}.torch'
            test_path = os.path.join(embeddings_parent_directory, local_name)
            local_name = f'images_cifar100_val_{model_name}.torch'
            val_path = os.path.join(embeddings_parent_directory, local_name)
            
        elif dataset == "cub":
            local_name = f'images_CUB_train_{model_name}.torch'
            train_path = os.path.join(embeddings_parent_directory, local_name)
            local_name = f'images_CUB_test_{model_name}.torch'
            test_path = os.path.join(embeddings_parent_directory, local_name)
            local_name = f'images_CUB_val_{model_name}.torch'
            val_path = os.path.join(embeddings_parent_directory, local_name)
        
        elif dataset == "climateTV":
            local_name = f'images_ClimateTV_train_{model_name}.torch'
            train_path = os.path.join(embeddings_parent_directory, local_name)
            local_name = f'images_ClimateTV_test_{model_name}.torch'
            test_path = os.path.join(embeddings_parent_directory, local_name)
            local_name = f'images_ClimateTV_val_{model_name}.torch'
            val_path = os.path.join(embeddings_parent_directory, local_name)
        
        elif dataset == "mit_states":
            local_name = f'images_MiT-States_train_{model_name}.torch'
            train_path = os.path.join(embeddings_parent_directory, local_name)
            local_name = f'images_MiT-States_test_{model_name}.torch'
            test_path = os.path.join(embeddings_parent_directory, local_name)
            local_name = f'images_MiT-States_val_{model_name}.torch'
            val_path = os.path.join(embeddings_parent_directory, local_name)
            
        elif dataset == "imagewoof":
            local_name = f'images_ImageWoof_train_{model_name}.torch'
            train_path = os.path.join(embeddings_parent_directory, local_name)
            local_name = f'images_ImageWoof_test_{model_name}.torch'
            test_path = os.path.join(embeddings_parent_directory, local_name)
            local_name = f'images_ImageWoof_val_{model_name}.torch'
            val_path = os.path.join(embeddings_parent_directory, local_name)
            
        elif dataset == "imagenette":
            local_name = f'images_ImageNette_train_{model_name}.torch'
            train_path = os.path.join(embeddings_parent_directory, local_name)
            local_name = f'images_ImageNette_test_{model_name}.torch'
            test_path = os.path.join(embeddings_parent_directory, local_name)
            local_name = f'images_ImageNette_val_{model_name}.torch'
            val_path = os.path.join(embeddings_parent_directory, local_name)
            
        elif dataset == "own_dataset": # TODO add own dataset and name paths accordingly
            local_name = f'images_own_dataset_{model_name}.torch'
            train_path = None
            val_path = None
            test_path = None
            
        # check if paths are valid
        if train_path is not None:
            if not os.path.exists(train_path):
                print(f"Path {train_path} does not exist")
                train_path = None
            else:
                self.embeddings_path_train = train_path
                
        if val_path is not None: 
            if not os.path.exists(val_path):
                print(f"Path {val_path} does not exist")
                val_path = None
            else:
                self.embeddings_path_val = val_path
                
        if test_path is not None:
            if not os.path.exists(test_path):
                print(f"Path {test_path} does not exist")
                test_path = None
            else:
                self.embeddings_path_test = test_path
        
        if test_path_second is not None:
            if not os.path.exists(test_path_second):
                print(f"Path {test_path_second} does not exist")
                test_path_second = None
            else:
                self.embeddings_path_test_second = test_path_second
                
    def load_data(self):
        
        train_path = self.embeddings_path_train
        val_path = self.embeddings_path_val
        test_path = self.embeddings_path_test
        test_path_second = self.embeddings_path_test_second
        
        if train_path is not None:
            train_data = torch.load(train_path, weights_only=False)
            print(f"Train data loaded from {train_path}")
            self.data_train_raw = train_data
        if val_path is not None:
            val_data = torch.load(val_path, weights_only=False)
            print(f"Validation data loaded from {val_path}")
            self.data_val_raw = val_data
        if test_path is not None:
            test_data = torch.load(test_path, weights_only=False)
            self.data_test_raw = test_data
            print(f"Test data loaded from {test_path}")
        if test_path_second is not None:
            test_data_second = torch.load(test_path_second, weights_only=False)
            self.data_test_second_raw = test_data_second
            print(f"Second Test data loaded from {test_path_second}")
            
    def extract_data_labels(self):
        #print("Extracting data and labels...")
        def split_data(dict_data):
            data_list = []
            labels_list = []

            for key in dict_data.keys():
                data_list.append(np.array(dict_data[key][0].cpu().numpy()))
                labels_list.append(dict_data[key][1])

            labels_list = np.array(labels_list)
            data_list = np.array(data_list)
                
            #return as numpy arrays
            return data_list, labels_list
        
        train_data = self.data_train_raw
        val_data = self.data_val_raw
        test_data = self.data_test_raw
        data_test_second = self.data_test_second_raw
        
        if train_data is not None:
            train_data, train_labels = split_data(train_data)
            self.data_train = train_data
            self.data_train_labels_raw = train_labels
            
            self.train_labels_int = train_labels.flatten()
            #print("Train labels: ", self.train_labels_int)
            encoder = OneHotEncoder().fit(np.array(self.data_train_labels_raw).reshape(-1, 1))
            self.train_labels_one_hot = encoder.transform(np.array(self.data_train_labels_raw).reshape(-1, 1)).toarray()
            
            #check if it is a string
            if self.train_labels_int.dtype != 'int64':
                self.train_labels_int = np.argmax(self.train_labels_one_hot, axis=1)

            
        if val_data is not None:
            val_data, val_labels = split_data(val_data)
            
            self.data_val = val_data
            self.data_val_labels_raw = val_labels
            
            self.val_labels_int = val_labels.flatten()
            #print("Val labels: ", len(np.unique(self.data_val_labels_raw)))
            self.val_labels_one_hot = encoder.transform(np.array(self.data_val_labels_raw).reshape(-1, 1)).toarray()
            
            if self.val_labels_int.dtype != 'int64':
                self.val_labels_int = np.argmax(self.val_labels_one_hot, axis=1)
            
        if test_data is not None:
            test_data, test_labels = split_data(test_data)
            self.data_test = test_data
            self.data_test_labels_raw = test_labels
            
            self.test_labels_int = test_labels.flatten()
            self.test_labels_one_hot = encoder.transform(np.array(self.data_test_labels_raw).reshape(-1, 1)).toarray()
            
            if self.test_labels_int.dtype != 'int64':
                self.test_labels_int = np.argmax(self.test_labels_one_hot, axis=1)
                
        if data_test_second is not None:
            data_test_second, test_labels_second = split_data(data_test_second)
            self.data_test_second = data_test_second
            self.data_test_labels_raw_second = test_labels_second
            
            self.test_labels_int_second = test_labels_second.flatten()
            self.test_labels_one_hot_second = encoder.transform(np.array(self.data_test_labels_raw_second).reshape(-1, 1)).toarray()
            
            if self.test_labels_int_second.dtype != 'int64':
                self.test_labels_int_second = np.argmax(self.test_labels_one_hot_second, axis=1)
        
    def zero_shot(self, model, preprocess, device):
        # Split the data into train and test sets
        val_exists = False
        
        X_test, y_test = self.data_test, self.test_labels_int

        classes_text = self.class_labels
        classes_text = [f'a photo of a {i.replace("_", " ")}.' for i in classes_text]
        label_tokens = clip.tokenize(classes_text).to(device)
        
        with torch.no_grad():  # Disable gradient calculations
            label_embeddings = model.encode_text(label_tokens).cpu().numpy()  # Use encode_text for text features

        # Normalize the embeddings
        label_embeddings /= np.linalg.norm(label_embeddings, axis=1, keepdims=True)

        image_embeddings = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)
        
        if isinstance(image_embeddings, np.ndarray):
            image_embeddings = torch.tensor(image_embeddings, dtype=torch.float32)
        if isinstance(label_embeddings, np.ndarray):
            text_embeddings = torch.tensor(label_embeddings, dtype=torch.float32)

        # Calculate the cosine similarity between image and text embeddings
        similarity = torch.matmul(image_embeddings, text_embeddings.T)

        # Get the index of the most similar text (class) for each image
        predictions = torch.argmax(similarity, dim=1)
        test_accuracy = accuracy_score(y_test, predictions)
        print("Zero-shot Test Accuracy:", test_accuracy)
        
    def linear_probe(self):
        # Split the data into train and test sets
        val_exists = False
        
        
        X_train, X_test, y_train, y_test = self.data_train, self.data_test, self.train_labels_int, self.test_labels_int

        # Initialize and train the Logistic Regression classifier
        classifier = LogisticRegression(max_iter=1000, verbose=1)
        classifier.fit(X_train, y_train)

        # Make predictions on the test set
        y_test_pred = classifier.predict(X_test)
        y_train_pred = classifier.predict(X_train)

        # Calculate accuracy
        test_accuracy = accuracy_score(y_test, y_test_pred)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        print(f"Training Accuracy: {train_accuracy}")
        print(f"Test Accuracy: {test_accuracy}")

        datetime_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Save the accuracies to a text file
        path_save = f"./linear_probes/accuracy_results_{datetime_now}.txt"
        with open(path_save, 'w') as f:
            f.write(f"Dataset: {self.dataset}\n")
            f.write(f"Model: {self.clip_model}\n")
            f.write(f"Training Accuracy: {train_accuracy}\n")
            f.write(f"Test Accuracy: {test_accuracy}\n")
        
    def get_image_concept_names(self, concept_path, concept_per_class):
        concept_names = []
        
        # ---- TODO -----
        # Define own dataset
        
        if self.dataset == "imagenette":
            dataset_name = "ImageNette/ImageNette"
        if self.dataset == "cub":
            dataset_name = "CUB/CUB"
        if self.dataset == "imagewoof":
            dataset_name = "ImageWoof/ImageWoof"
        if self.dataset == "cifar10":
            dataset_name = "cifar10/cifar10"
        if self.dataset == "cifar100":
            dataset_name = "cifar100/cifar100"
        if self.dataset == "places365":
            dataset_name = "Places365/Places365"
        if self.dataset == "imagenet":
            dataset_name = "ImageNet/ImageNet"
        if self.dataset == "climateTV":
            dataset_name = "ClimateTV/ClimateTV"
        if self.dataset == "mit_states":
            dataset_name = "MiT-States/MiT-States"
        
        self.class_num = concept_per_class
        path_names = f'{concept_path}/{dataset_name}_rand_{concept_per_class}.txt'
        
        if os.path.exists(path_names):
            with open(path_names, 'r') as f:
                concept_names = [line.strip().split('.')[0] for line in f.readlines()]
        return concept_names
    
    def load_concepts(self, segment_path, segmentation_technique, concept_name, selected_image_concepts = None, concept_per_class = None, crop = False):
    

        model_name = self.clip_model
        embeddings_parent_directory = segment_path
        dataset = self.dataset
        seg_tech = segmentation_technique
        self.segmentation_technique = seg_tech
        self.concept_name = concept_name
        
        # ---- TODO -----
        # Define own dataset
        
        if dataset == "cub":
            if not crop:
                if concept_name != None:
                    concepts = f'segcrop_CUB_200_2011_{seg_tech}_{concept_name}_{model_name}.torch'
                else:
                    concepts = f'segcrop_CUB_200_2011_{seg_tech}_{model_name}.torch'
            else:
                self.background = True
                if concept_name != None:
                    concepts = f'segmask_CUB_200_2011_{seg_tech}_{concept_name}_{model_name}.torch'
                else:
                    concepts = f'segmask_CUB_200_2011_{seg_tech}_{model_name}.torch'
            
        elif dataset == "imagenette":
            if not crop:
                if concept_name != None:
                    concepts = f'segcrop_ImageNette_{seg_tech}_{concept_name}_{model_name}.torch'
                else:
                    concepts = f'segcrop_ImageNette_{seg_tech}_{model_name}.torch'
            else:
                self.background = True
                if concept_name != None:
                    concepts = f'segmask_ImageNette_{seg_tech}_{concept_name}_{model_name}.torch'
                else:
                    concepts = f'segmask_ImageNette_{seg_tech}_{model_name}.torch'
            
        elif dataset == "imagewoof":
            if not crop:
                if concept_name != None:
                    concepts = f'segcrop_ImageWoof_{seg_tech}_{concept_name}_{model_name}.torch'
                else:
                    concepts = f'segcrop_ImageWoof_{seg_tech}_{model_name}.torch'
            else:
                self.background = True
                if concept_name != None:
                    concepts = f'segmask_ImageWoof_{seg_tech}_{concept_name}_{model_name}.torch'
                else:
                    concepts = f'segmask_ImageWoof_{seg_tech}_{model_name}.torch'
                    
        elif dataset == "cifar10":
            if not crop:
                if concept_name != None:
                    concepts = f'segcrop_cifar10_{seg_tech}_{concept_name}_{model_name}.torch'
                else:
                    concepts = f'segcrop_cifar10_{seg_tech}_{model_name}.torch'
            else:
                self.background = True
                if concept_name != None:
                    concepts = f'segmask_ifar10_{seg_tech}_{concept_name}_{model_name}.torch'
                else:
                    concepts = f'segmask_cifar10_{seg_tech}_{model_name}.torch'
                    
        elif dataset == "cifar100":
            if not crop:
                if concept_name != None:
                    concepts = f'segcrop_cifar100_{seg_tech}_{concept_name}_{model_name}.torch'
                else:
                    concepts = f'segcrop_cifar100_{seg_tech}_{model_name}.torch'
            else:
                self.background = True
                if concept_name != None:
                    concepts = f'segmask_cifar100_{seg_tech}_{concept_name}_{model_name}.torch'
                else:
                    concepts = f'segmask_cifar100_{seg_åtech}_{model_name}.torch'
                    
        elif dataset == "places365":
            if not crop:
                if concept_name != None:
                    concepts = f'segcrop_Places365_{seg_tech}_{concept_name}_{model_name}.torch'
                else:
                    concepts = f'segcrop_Places365_{seg_tech}_{model_name}.torch'
            else:
                self.background = True
                if concept_name != None:
                    concepts = f'segmask_Places365_{seg_tech}_{concept_name}_{model_name}.torch'
                else:
                    concepts = f'segmask_Places365_{seg_tech}_{model_name}.torch'
                    
        elif dataset == "mit_states":
            if not crop:
                if concept_name != None:
                    concepts = f'segcrop_MiT-States_{seg_tech}_{concept_name}_{model_name}.torch'
                else:
                    concepts = f'segcrop_MiT-States_{seg_tech}_{model_name}.torch'
            else:
                self.background = True
                if concept_name != None:
                    concepts = f'segmask_MiT-States_{seg_tech}_{concept_name}_{model_name}.torch'
                else:
                    concepts = f'segmask_MiT-States_{seg_tech}_{model_name}.torch'
                    
        elif dataset == "imagenet":
            if not crop:
                if concept_name != None:
                    concepts = f'segcrop_ImageNet_{seg_tech}_{concept_name}_{model_name}.torch'
                else:
                    concepts = f'segcrop_ImageNet_{seg_tech}_{model_name}.torch'
            else:
                self.background = True
                if concept_name != None:
                    concepts = f'segmask_ImageNet_{seg_tech}_{concept_name}_{model_name}.torch'
                else:
                    concepts = f'segmask_ImageNet_{seg_tech}_{model_name}.torch'
        
        elif dataset == "climateTV":
            if not crop:
                if concept_name != None:
                    concepts = f'segcrop_ClimateTV_{seg_tech}_{concept_name}_{model_name}.torch'
                else:
                    concepts = f'segcrop_ClimateTV_{seg_tech}_{model_name}.torch'
            else:
                self.background = True
                if concept_name != None:
                    concepts = f'segmask_ClimateTV_{seg_tech}_{concept_name}_{model_name}.torch'
                else:
                    concepts = f'segmask_ClimateTV_{seg_tech}_{model_name}.torch'
        
        concept_path = os.path.join(embeddings_parent_directory, concepts)
        concepts_dict = torch.load(concept_path) 
        
        print("Concepts loaded from ", concept_path)

        if selected_image_concepts is not None and concept_per_class is not None:
            #print("concept_image_names", concept_image_names)
            if self.dataset == "imagenet":
                concept_image_names = self.get_image_concept_names(selected_image_concepts, concept_per_class)

                if os.path.exists(f"./processed_data/{self.dataset}_selected_classes_{seg_tech}_{crop}_{concept_name}_{model_name}_{concept_per_class}.torch"):
                    concepts_dict = torch.load(f"./processed_data/{self.dataset}_selected_classes_{seg_tech}_{crop}_{concept_name}_{model_name}_{concept_per_class}.torch")
                else:
                    concept_dict = {}
                    for key in tqdm(concept_image_names):
                        for segment_key in concepts_dict.keys():
                            if key in segment_key:
                                concept_dict[segment_key] = concepts_dict[segment_key]
                                break
                    torch.save(concept_dict, f"./processed_data/{self.dataset}_selected_classes_{seg_tech}_{crop}_{concept_name}_{model_name}_{concept_per_class}.torch")
                    concepts_dict = concept_dict
            else:
                concept_image_names = set(self.get_image_concept_names(selected_image_concepts, concept_per_class))
                concept_dict = {}
                for key in tqdm(concept_image_names):
                    for segment_key in concepts_dict:
                        if key in segment_key:
                            concept_dict[segment_key] = concepts_dict[segment_key]
                            break
            
                concepts_dict = concept_dict
        keys = list(concepts_dict.keys())
        concepts_np = []
        for key in keys:
            local_emb = concepts_dict[key][0].cpu().numpy()
            if len(local_emb.shape)> 0:
                concepts_np.append(local_emb)
        concepts_np = np.array(concepts_np)
        self.image_segments = concepts_np
        self.image_segments_names = keys
        
    def cluster_image_concepts(self, clustering_method='kmeans', n_clusters=10, pca=False, **kwargs):
        # set seed for reproducibility
        print("PCA: ", pca)
        # Initialize the clustering model based on the method
        
        norm = np.linalg.norm(self.image_segments.copy(), axis=1, keepdims=True)
        image_embedding = self.image_segments.copy() / norm
        
        if n_clusters > image_embedding.shape[0]:
            n_clusters = image_embedding.shape[0]
        
        print("Number of image embeddings: ", image_embedding.shape[0])
        print("Number of clusters: ", n_clusters)
        
        if pca:
            pca = PCA(n_components=100)  # You can adjust the number of components as needed
            image_embedding_pca = pca.fit_transform(image_embedding)
        else:
            image_embedding_pca = image_embedding
        
        self.cluster_technique = clustering_method
        print("Clustering method: ", clustering_method)
        if clustering_method == 'kmeans':
            clustering_model = KMeans(n_clusters=n_clusters, random_state=42, )
            cluster_labels = clustering_model.fit_predict(image_embedding_pca)
        elif clustering_method == 'hierarchical':
            clustering_model = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
            cluster_labels = clustering_model.fit_predict(image_embedding_pca)
        elif clustering_method == 'dbscan':
            clustering_model = DBSCAN(**kwargs)
            cluster_labels = clustering_model.fit_predict(image_embedding_pca)
        else:
            raise ValueError(f"Unsupported clustering method: {clustering_method}")
        # create dicts with cluster ids and empty lists
        clustered_concepts = {
            i: [] for i in np.unique(cluster_labels) if i != -1  # Exclude noise in DBSCAN
        }
                
        clustered_images = {
            i: [] for i in np.unique(cluster_labels) if i != -1  # Exclude noise in DBSCAN
        }
        # fill dict with embeddings and file names
        for i, label in enumerate(cluster_labels):
            if label != -1:  # Handle noise for DBSCAN
                clustered_concepts[label].append(self.image_segments[i])
                clustered_images[label].append(self.image_segments_names[i])
                
        self.clustered_concepts_all = clustered_concepts
        self.image_segments_names = clustered_images

        if self.dataset == "cub":
            dataset = "CUB_200_2011"
        if self.dataset == "places365":
            dataset = "Places365"

        # save intermediate output
        if self.to_save_concepts:
            if self.concept_name is not None:
                torch.save(clustered_images, f"./clusters/{dataset}/clustered_images_{dataset}_{self.segmentation_technique}_{self.clip_model}_{self.concept_name}_{self.class_num}_{n_clusters}.torch")
            else:
                torch.save(clustered_images, f"./clusters/{dataset}/clustered_images_{dataset}_{self.segmentation_technique}_{self.clip_model}_X_{self.class_num}_{n_clusters}.torch")

    def centroid_concepts(self, centroid_method='mean'):
        res = {}
        self.centroid_method = centroid_method
        clustered_concepts = self.clustered_concepts_all.copy()
        n_clusters = len(clustered_concepts)
        # Compute mean of the clustered concepts
        if centroid_method == 'mean':
            for key in clustered_concepts.keys():
                clustered_concepts[key] = np.mean(clustered_concepts[key], axis=0)
        elif centroid_method == 'median':
            for key in clustered_concepts.keys():
                median_values = np.median(clustered_concepts[key], axis=0)
                distances = np.sum(np.abs(clustered_concepts[key] - median_values), axis=1)
                median_row_index = np.argmin(distances)
                clustered_concepts[key] = clustered_concepts[key][median_row_index]
                res[self.image_segments_names[key][median_row_index]] = clustered_concepts[key]
                
        if self.dataset == "cub":
            dataset = "CUB_200_2011"
        if self.dataset == "places365":
            dataset = "Places365"
            
        if self.to_save_concepts:
            if self.concept_name is not None:
                torch.save(clustered_concepts, f"./clusters/{dataset}/cluster_centroids_{dataset}_{self.segmentation_technique}_{self.clip_model}_{self.concept_name}_{self.class_num}_{n_clusters}.torch")
                torch.save(res, f"./clusters/{dataset}/cluster_centroids_named_{self.dataset}_{self.segmentation_technique}_{self.clip_model}_{self.concept_name}_{self.class_num}_{n_clusters}_.torch")

            else:
                torch.save(clustered_concepts, f"./clusters/{dataset}/cluster_centroids_{dataset}_{self.segmentation_technique}_{self.clip_model}_X_{self.class_num}_{n_clusters}.torch")
                torch.save(res, f"./clusters/{dataset}/cluster_centroids_named_{dataset}_{self.segmentation_technique}_{self.clip_model}_X_{self.class_num}_{n_clusters}.torch")
            
        # Convert the final cluster concepts back to an array
        clustered_concepts = np.array([clustered_concepts[key] for key in clustered_concepts.keys()])
        self.clustered_concepts = clustered_concepts
        
    def concept_intervention(self, concepts_remove, similarity_threshold=0.5):
        #remove in  self.clustered_concepts all concepts in concepts_remove that are above the similarity threshold
        
        self.clustered_concepts = self.full_concepts.copy()
        
        # TODO: change backbone if not CLIP or different version of CLIP
        if self.clip_model == "CLIP-ViT-L14":        
            model, preprocess = clip.load("ViT-L/14", device="cuda", jit=False)
        elif self.clip_model == "CLIP-ViT-B16":
            model, preprocess = clip.load("ViT-B/15", device="cuda", jit=False)
        elif self.clip_model == "resnet50":
            model, preprocess = clip.load("RN50", device="cuda", jit=False)

        tokenized_text = clip.tokenize(concepts_remove).to("cuda")
        #embed concepts_remove with clip text
        with torch.no_grad():
            concepts_remove_embedding = model.encode_text(tokenized_text).detach().cpu().numpy()
        
        similarities = torch.nn.functional.cosine_similarity(torch.tensor(concepts_remove_embedding), torch.tensor(self.clustered_concepts), dim=1)
        
        # Find the indices of the concepts to be removed that are above the similarity threshold
        indices_remove = np.where(similarities > similarity_threshold)[0]
        
        # Remove the concepts from the clustered concepts
        self.clustered_concepts = np.delete(self.clustered_concepts, indices_remove, axis=0)
        print(f"Removed {len(indices_remove)} concepts from the clustered concepts.")
    
    
    def preprocess_module(self, data, concepts, batch_size=1000):
        if concepts is not None:
            device = self.device
            # Ensure data and concepts are 2D arrays
            data = np.array(data).reshape(-1, data.shape[-1])  # Reshape in case data is 1D
            concepts = np.array(concepts).reshape(-1, concepts.shape[-1])  # Reshape in case concepts is 1D

            # Convert concepts to a torch tensor and transfer to the GPU
            concepts = torch.tensor(concepts, device=device, dtype=torch.float32)
            norms_c_i_squared = torch.norm(concepts, dim=1, keepdim=True) ** 2

            def process_batch(batch):
                # Move the batch to the GPU
                batch = torch.tensor(batch, device=device, dtype=torch.float32)
                
                # Calculate dot products and normalization for each item in the batch
                multiplied_image_batch = torch.matmul(batch, concepts.T) / norms_c_i_squared.T
                
                # Apply softmax on each row in the batch using torch's softmax with dim argument
                multiplied_image_softmax_batch = torch.nn.functional.softmax(multiplied_image_batch, dim=1)
                return multiplied_image_softmax_batch.cpu().numpy()  # Move back to CPU for concatenation

            num_batches = int(np.ceil(len(data) / batch_size))
            data_batches = [data[i*batch_size : (i+1)*batch_size] for i in range(num_batches)]
            
            # Process batches on the GPU, and retrieve results
            results = []
            for batch in tqdm(data_batches, desc="Processing Batches"):
                results.append(process_batch(batch))

            # Concatenate results from all batches
            multiplied_image_softmax = np.vstack(results)
        else:
            multiplied_image_softmax = data
        return multiplied_image_softmax
    
    def preprocess_data(self, type_ = None, label_type = False):
        
            
        #print("Preprocess data...")
        multiplied_image_softmax_train = self.preprocess_module(self.data_train.copy(), self.clustered_concepts.copy())
                    
        X_test, X_val = None, None
        
        if self.standard_scaler:
            self.scaler = StandardScaler()
            self.scaler.fit(multiplied_image_softmax_train)
            X_train = self.scaler.transform(multiplied_image_softmax_train)
            if label_type:
                y_train = self.train_labels_int
            else:
                y_train = self.train_labels_one_hot
                
        if self.data_val is not None:
            multiplied_image_softmax_val = self.preprocess_module(self.data_val.copy(), self.clustered_concepts.copy())
            X_val = self.scaler.transform(multiplied_image_softmax_val)
            if label_type:
                y_val = self.val_labels_int
            else:
                y_val = self.val_labels_one_hot
        else:
            print("No validation data")

        if self.data_test is not None:
            multiplied_image_softmax_test = self.preprocess_module(self.data_test.copy(), self.clustered_concepts.copy())
            X_test = self.scaler.transform(multiplied_image_softmax_test)  
            if label_type:
                y_test = self.test_labels_int
            else:
                y_test = self.test_labels_one_hot     
        else:
            print("No test data")
            
        if self.data_test_second is not None:
            multiplied_image_softmax_test_second = self.preprocess_module(self.data_test_second.copy(), self.clustered_concepts.copy())
            X_test_second = self.scaler.transform(multiplied_image_softmax_test_second)  
            if label_type:
                y_test_second = self.test_labels_int_second
            else:
                y_test_second = self.test_labels_one_hot_second
                
        print(X_train.shape, X_test.shape, X_val.shape, y_train.shape, y_test.shape, y_val.shape)
            
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test  
        if self.data_test_second is not None:
            self.y_test_second = y_test_second
            self.X_test_second = X_test_second 

    def train(self, 
            lambda_1=1e-4,  # L1 regularization weight # original 1e-4
            lr=1e-3,  # original 1e-3
            num_epochs=10, 
            batch_size=32, 
            project = "SegCBM_50",
            device='cpu', 
            to_print=False, 
            use_wandb=False, 
            early_stopping_patience=None,
            one_hot=False):
        print("learning rate: ", lr)
        print("lambda_1: ", lambda_1)
        """
        Train the Concept Bottleneck Model (CBM) with cross-entropy loss, L1 regularization (for sparsity),
        L2 regularization (for weight decay), and validation-based tuning.

        Args:
            lambda_1: Hyperparameter for L1 regularization to encourage sparsity.
            lr: Learning rate.
            num_epochs: Number of epochs to train the model.
            batch_size: Batch size.
            device: Device to run the training ('cuda' or 'cpu').
            to_print: Whether to print progress.
            early_stopping_patience: Number of epochs to wait for improvement in validation loss before stopping early.
            one_hot: Boolean indicating if labels are one-hot encoded.
        """
        
        def cross_entropy_one_hot(input, target):
            """
            Custom cross-entropy loss function that accepts one-hot encoded labels.

            Args:
                input: Logits from the model (before softmax).
                target: One-hot encoded labels.

            Returns:
                Loss value.
            """
            log_prob = F.log_softmax(input, dim=1)
            loss = -torch.sum(target * log_prob, dim=1).mean()
            return loss
        
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{self.dataset}_{self.clip_model}_{current_time}"
        # Initialize Weights & Biases (W&B) with run name
        if use_wandb:
            wandb.init(project=project, name=run_name)
            wandb.log({'lambda_1': lambda_1, 
                    'one_hot': one_hot,
                    'lr': lr, 
                    'batch_size': batch_size,
                    "segmentation_technique": self.segmentation_technique, 
                    "concept_name": self.concept_name,
                    "clusters": self.clustered_concepts.shape[0],
                    "cluster_technique": self.cluster_technique,
                    "centroid_method": self.centroid_method,
                    "model_name": self.clip_model,
                    "Max_class_num": self.class_num,
                    "background": self.background})
        
        class LinearProbe(nn.Module):
            def __init__(self, input_dim, num_classes):
                super(LinearProbe, self).__init__()
                self.linear = nn.Linear(input_dim, num_classes)

            def forward(self, x):
                return self.linear(x)
        
        # Determine number of classes based on labels
        if not one_hot:
            num_classes = self.y_train.shape[1]
        else:
            num_classes = len(np.unique(self.y_train))
        
        model = LinearProbe(self.X_train.shape[1], num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Choose the appropriate criterion
        if not one_hot:
            criterion = cross_entropy_one_hot
        else:
            criterion = nn.CrossEntropyLoss()
        self.criterion = criterion

        # Convert data to tensors
        X_train = torch.tensor(self.X_train, dtype=torch.float32).to(device)
        X_test = torch.tensor(self.X_test, dtype=torch.float32).to(device)
        X_val = torch.tensor(self.X_val, dtype=torch.float32).to(device)

        if not one_hot:
            # Labels are one-hot encoded
            y_train = torch.tensor(self.y_train, dtype=torch.float32).to(device)
            y_test = torch.tensor(self.y_test, dtype=torch.float32).to(device)
            y_val = torch.tensor(self.y_val, dtype=torch.float32).to(device)

        else:
            # Labels are class indices
            y_train = torch.tensor(self.y_train, dtype=torch.long).to(device)
            y_test = torch.tensor(self.y_test, dtype=torch.long).to(device)
            y_val = torch.tensor(self.y_val, dtype=torch.long).to(device)
                
        # Create TensorDataset and DataLoader for training data
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Create DataLoader for test data
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # Create DataLoader for validation data if available
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


        best_val_loss = float('inf')
        best_model_weights = model.state_dict()
        patience_counter = 0

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            total_l1_loss = 0.0
            total_ce_loss = 0.0
            total_samples = 0
            correct_predictions = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)

                # Compute the cross-entropy loss
                if not one_hot:
                    ce_loss = criterion(outputs, batch_y)
                    _, predicted = torch.max(outputs, 1)
                    _, true_labels = torch.max(batch_y, 1)
                else:
                    ce_loss = criterion(outputs, batch_y)
                    _, predicted = torch.max(outputs, 1)
                    true_labels = batch_y

                # L1 regularization for sparsity
                l1_loss = lambda_1 * torch.norm(model.linear.weight, p=1)

                # Total loss
                loss = ce_loss + l1_loss

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_X.size(0)
                total_l1_loss += l1_loss.item() * batch_X.size(0)
                total_ce_loss += ce_loss.item() * batch_X.size(0)
                total_samples += batch_X.size(0)
                correct_predictions += (predicted == true_labels).sum().item()

            # Calculate average losses and accuracy
            avg_train_loss = running_loss / total_samples
            avg_l1_loss = total_l1_loss / total_samples
            avg_ce_loss = total_ce_loss / total_samples
            train_accuracy = 100 * correct_predictions / total_samples

            if use_wandb:
                wandb.log({
                    'train_loss': avg_train_loss,
                    'train_accuracy': train_accuracy,
                    'l1_loss': avg_l1_loss,
                    'ce_loss': avg_ce_loss
                }, step=epoch+1)

            self.model = model
            # Evaluate on test set
            test_loss, test_accuracy = self.evaluate(test_loader, device, one_hot)
            if use_wandb:
                wandb.log({'test_loss': test_loss, 'test_accuracy': test_accuracy}, step=epoch+1)

            # Evaluate on validation set if available
            val_loss, val_accuracy = self.evaluate(val_loader, device, one_hot)
            if use_wandb:
                wandb.log({'val_loss': val_loss, 'val_accuracy': val_accuracy}, step=epoch+1)

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                if to_print:
                    print("Early stopping due to no improvement in validation loss.")
                break

            if to_print:
                print(f"Epoch [{epoch+1}/{num_epochs}] | "
                    f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
                    f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.2f}%")
                if val_loader is not None:
                    print(f"Validation Loss: {val_loss:.4f} | Validation Acc: {val_accuracy:.2f}%")


        model.load_state_dict(best_model_weights)
        self.model = model
        if use_wandb:
            wandb.finish()

        return model
    
    
    def plot_instance_feature_importance(self, index, n=5, save = False):
        """
        Plot the top n features contributing to the model's prediction for a given instance.

        Args:
            instance: A single instance (feature vector) as a numpy array or torch tensor.
            n: Number of top features to plot.
        """
        # Ensure the model is trained
        if not hasattr(self, 'model'):
            raise ValueError("Model not trained yet. Please train the model before plotting feature importance.")

        instance = self.X_test[index:index+1]
        # Convert instance to tensor if it's a numpy array
        if isinstance(instance, np.ndarray):
            instance = torch.tensor(instance, dtype=torch.float32)

        # Move instance to appropriate device
        device = next(self.model.parameters()).device
        instance = instance.to(device)

        # Add batch dimension if necessary
        if instance.dim() == 1:
            instance = instance.unsqueeze(0)

        # Get the weights and biases from the trained model
        weights = self.model.linear.weight.detach().cpu().numpy()  # Shape: (num_classes, num_features)
        biases = self.model.linear.bias.detach().cpu().numpy()     # Shape: (num_classes,)

        # Get the feature values
        feature_values = instance.detach().cpu().numpy().squeeze()  # Shape: (num_features,)

        # Compute the contribution of each feature to each class
        # For linear models: contribution = weight * feature_value
        contributions = weights * feature_values  # Shape: (num_classes, num_features)

        # Get the predicted class
        with torch.no_grad():
            outputs = self.model(instance)
            probabilities = torch.softmax(outputs, dim=1).squeeze()
            predicted_class = torch.argmax(probabilities).item()

        # Get the contributions for the predicted class
        class_contributions = contributions[predicted_class]

        # Sort the features by absolute contribution
        sorted_indices = np.argsort(np.abs(class_contributions))[::-1]
        top_indices = sorted_indices[:n]
        top_contributions = class_contributions[top_indices]
        
        # Get feature names if available
        if hasattr(self, 'feature_names'):
            top_features = [self.feature_names[i] for i in top_indices]
        else:
            top_features = [str(i) for i in top_indices]

        # Plot the top features
        plt.figure(figsize=(10, 5))
        colors = ['red' if c < 0 else 'blue' for c in top_contributions]
        plt.bar(range(n), top_contributions, color=colors, tick_label=top_features)
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Top {n} Feature Contributions for Predicted Class {predicted_class}")
        plt.xlabel("Feature")
        plt.ylabel("Contribution")
        plt.tight_layout()
        plt.show()
        
        print(f"True Class: {self.class_labels[np.argmax(self.y_test[index:index+1][0])]}")
        print(f"Predicted Class: {self.class_labels[predicted_class]}")
        
        if save:
            plt.savefig(f"./imgs/feature_importance_{index}.pdf")
        return top_features, top_contributions
    
    def evaluate(self, data_loader, device, one_hot=False):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                outputs = self.model(batch_X)

                # Compute the loss
                loss = self.criterion(outputs, batch_y)

                # Accumulate the total loss
                total_loss += loss.item() * batch_X.size(0)

                # Predictions and true labels
                _, predicted = torch.max(outputs, 1)
                if not one_hot:
                    _, true_labels = torch.max(batch_y, 1)
                else:
                    true_labels = batch_y

                # Update total and correct predictions
                total += batch_X.size(0)
                correct += (predicted == true_labels).sum().item()

        # Calculate average loss and accuracy
        avg_loss = total_loss / total
        accuracy = 100 * correct / total
        return avg_loss, accuracy
    
class CBM_Model:

    def __init__(self, model, concepts, concept_processing, scaler, device = "cpu"):

        self.model = model
        self.concepts = concepts
        self.concept_processing = concept_processing
        self.scaler = scaler
        self.device = device

    def predict(self, data):
        data = self.concept_processing("standard", data, self.concepts)
        if self.scaler != None:
            data = self.scaler.transform(data)
        data = torch.tensor(data, dtype=torch.float32).to(self.device)
        self.model = self.model.to(self.device).to(torch.float32)
        predictions = self.model(data)
        predictions = predictions.detach().cpu().numpy()
        predictions = np.argmax(predictions, axis=1)
        return predictions

    def predict_processed(self, data):
        # Ensure data type consistency
        data = torch.tensor(data, dtype=torch.float32).to(self.device)
        self.model = self.model.to(self.device).to(torch.float32)

        predictions = self.model(data)
        predictions = predictions.detach().cpu().numpy()
        predictions = np.argmax(predictions, axis=1)
        return predictions
    
