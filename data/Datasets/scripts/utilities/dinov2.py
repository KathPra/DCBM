# load packages
import torch
from PIL import Image, ImageFile
import os
import tqdm
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Function to compute and save embeddings
def compute_embeddings(img_dir, model, transform, output_file, device):
    embeddings = {}
    for image_path in tqdm.tqdm(img_dir):
        image_name = image_path.split("/")[-1]

        # load image
        image = Image.open(image_path)

        # Handle images with transparency
        if image.mode in ('P', 'RGBA'): image = image.convert("RGBA")
        else: image = image.convert("RGB")

        # use clip required preprocessing
        image = transform(image).unsqueeze(0)  # Add batch dimension
        image = image.to(device)#.unsqueeze(0)
        with torch.no_grad():
            embedding = model(image).detach().cpu()#.numpy()  # Remove batch dimension and convert to numpy
        #print(output.shape)
        #embedding = output.last_hidden_state[-1].detach().cpu().numpy()

        #print(embedding.shape)
        class_name = image_path.split('/')[-2]
        embeddings[image_name] = [embedding, class_name]
    torch.save(embeddings, output_file)


# Function to compute and save embeddings as a dictionary
def compute_embeddings_batched(image_paths, model, transform, output_file, device, batch_size=64):

    embeddings = {}
    counter = 0
    
    # Process images in batches
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(image_paths), batch_size), desc="Embedding Segments"):
            batch_images = []
            batch_keys = image_paths[i:i+batch_size]

            for x in range(i, min(i + batch_size, len(image_paths))):
                image_path = image_paths[i]

                # Load image
                image = Image.open(image_path)

                # Handle images with transparency
                if image.mode in ('P', 'RGBA'):
                    image = image.convert("RGBA")
                else:
                    image = image.convert("RGB")

                # Preprocess image for the model
                batch_images.append(transform(image).unsqueeze(0))  # Add batch dimension

            # Combine batch images into a tensor and move to device
            batch_images = torch.cat(batch_images).to(device)
            
            # Compute embeddings for the batch
            batch_embeddings = model(batch_images).detach().cpu()#.numpy() # Move back to CPU
            # Store each embedding in the dictionary with the corresponding image key
            for idx, image_path in enumerate(batch_keys):
                image_name = image_path.split("/")[-1]
                class_name = image_path.split("/")[-2]
                try:
                    embeddings[image_name] = [batch_embeddings[idx], class_name]
                except Exception as e:
                    print(f"Error saving {image_name}: {e}")
                    pass

            if len(embeddings) > 100000:
                # Save the embeddings dictionary to a file
                outfile = output_file.replace(".torch", f"_{counter}.torch")
                torch.save(embeddings, outfile)
                print(f"Saved embeddings dictionary to {outfile}")
                embeddings = {}
                counter += 1

        # Save the remaining embeddings
        if embeddings:  # If there are any remaining embeddings to save
            outfile = output_file if counter == 0 else output_file.replace(".torch", f"_{counter}.torch")
            torch.save(embeddings, outfile)
            print(f"Saved embeddings dictionary to {outfile} ({len(embeddings)} embeddings)")

 
# Function to process a batch of images and compute embeddings
def process_batch(image_paths, model, transform, device):
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
 
        # Handle images with transparency
        if image.mode in ('P', 'RGBA'):
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")
 
        # Apply the CLIP required preprocessing
        images.append(transform(image).unsqueeze(0))  # Add batch dimension
    
    # Stack images into a single batch tensor
    images = torch.cat(images).to(device)
    
    # Perform inference
    with torch.no_grad():
        embeddings = model.encode_image(images)
 
    return embeddings.cpu()
 
# Function to compute and save embeddings for a folder using batch processing
def compute_folder_embeddings_batch(folder_name, model, transform, device, output_local_file, batch_size=150):
    embeddings_local = {}
    output_local = os.path.join(output_local_file, folder_name.split('/')[-1] + '.torch')
    
    if not os.path.exists(output_local):
        print("Creating embeddings for:", folder_name)
        image_paths = [os.path.join(folder_name, image_name) for image_name in os.listdir(folder_name)]
        
        # Process images in batches
        for i in tqdm.tqdm(range(0, len(image_paths), batch_size)):
            batch_image_paths = image_paths[i:i+batch_size]
            embeddings = process_batch(batch_image_paths, model, transform, device)
 
            # Store embeddings with corresponding image names
            for j, image_path in enumerate(batch_image_paths):
                image_name = folder_name.split('/')[-1] + '_' + os.path.basename(image_path)
                embeddings_local[image_name] = embeddings[j]
        
        torch.save(embeddings_local, output_local)
        print("Embeddings saved for:", folder_name)
    else:
        print("Embedding already exists for:", folder_name)
 
# Function to parallelize folder embeddings with batch processing
def compute_embeddings_parallel(img_dirs, model, transform, output_file, device, batch_size=100):
    output_local_file = "./Embeddings/"
    if not os.path.exists(output_local_file):
        os.makedirs(output_local_file)
    
    # Parallelize embedding computation using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_folder_embeddings_batch, folder_name, model, transform, device, output_local_file, batch_size)
                   for folder_name in img_dirs]
 
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            future.result()  # Ensure any exceptions are raised
 
    # Combine the embeddings into a single file
    output_file_ = "./Embeddings/images_incomplete.torch"
    if os.path.exists(output_file_):
        combined_embeddings = torch.load(output_file_)
        used_keys = torch.load("./Embeddings/used_keys.torch")
    else:
        used_keys = []
        combined_embeddings = {}
    for file in glob.glob(output_local_file + "*"):
        #embeddings_local = torch.load(file)
        #combined_embeddings.update(embeddings_local)
        try:
            if not file in used_keys:
                print(file)
                embeddings_local = torch.load(file)
                combined_embeddings.update(embeddings_local)
                used_keys.append(file)
        except Exception as e:
            print(f"Error saving {file}: {e}")
            if os.path.exists(file):
                os.remove(file)
            torch.save(combined_embeddings, output_file_)
            torch.save(used_keys, "./Embeddings/used_keys.torch")
            raise
            pass
    torch.save(combined_embeddings, output_file)
    # delete all entries in output_local_file
    for file in glob.glob(output_local_file + "*"):
        os.remove(file)
    print(f"Total embeddings: {len(combined_embeddings.keys())}")
    return combined_embeddings
 
 
# Function to compute and save embeddings
def compute_embeddings_pt(img_dir, model, transform, output_file, device):
    embeddings = {}
    for image_name in tqdm.tqdm(img_dir):
        image_path = img_dir[image_name]

        # load image
        image = Image.open(image_path)
        if image.mode == 'RGBA' or image.mode == 'LA':  # LA for grayscale + alpha
            # Discard the alpha channel and convert to RGB
            image = image.convert("RGBA")
        elif image.mode == 'P':  # 'P' mode means palette-based image, convert to RGB
            image = image.convert("RGBA")
        else:
            # For all other modes, convert to RGB directly (handles grayscale or other modes)
            image = image.convert("RGB")
            # Handle images with transparency
            if image.mode in ('P', 'RGBA'): 
                image = image.convert("RGBA")
            else: image = image.convert("RGB")

        # use clip required preprocessing
        image = transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            embedding = model(image.to(device)).squeeze()#.numpy()  # Remove batch dimension and convert to numpy

        embeddings[image_name] = embedding
    
    torch.save(embeddings, output_file)

### copied from dinov2/dinov2/data/transforms.py (https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py)
# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean= (0.485, 0.456, 0.406),
    std= (0.229, 0.224, 0.225),
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)

# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)

def make_normalize_transform(
    mean= IMAGENET_DEFAULT_MEAN,
    std= IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)

#######################################