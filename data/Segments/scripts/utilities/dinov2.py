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
def compute_embeddings_batched(image_paths, model, transform, output_file, device, batch_size=248):

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