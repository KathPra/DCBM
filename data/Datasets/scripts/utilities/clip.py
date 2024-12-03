# load packages
import torch
from PIL import Image, ImageFile
import os
import clip
import tqdm
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Function to compute and save embeddings
def compute_embeddings(img_dir, model, transform, output_file, device):
    embeddings = {}
    for image_path in tqdm.tqdm(img_dir):
        image_name = image_path.split("/")[-1]

        # load class
        class_name = image_path.split('/')[-2]
        # load image
        image = Image.open(image_path)

        # Handle images with transparency
        if image.mode in ('P', 'RGBA'): image = image.convert("RGBA")
        else: image = image.convert("RGB")

        # use clip required preprocessing
        image = transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            embedding = model.encode_image(image.to(device)).squeeze()#.numpy()  # Remove batch dimension and convert to numpy

        embeddings[image_name] = [embedding,class_name]
    
    torch.save(embeddings, output_file)


# Function to compute and save embeddings as a dictionary
def compute_embeddings_batched(image_paths, model, transform, output_file, device, batch_size=64):
    
    embeddings = {}
    counter = 0
    
    # Process images in batches
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(image_paths), batch_size), desc="Embedding Images"):
            batch_images = []
            batch_keys = image_keys[i:i+batch_size]

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
            batch_embeddings = model.encode_image(batch_images).cpu()  # Move back to CPU

            # Store each embedding in the dictionary with the corresponding image key
            for idx, image_name in enumerate(batch_keys):
                image_name = image_path.split("/")[-1]
                class_name = image_path.split("/")[-2]
                try:
                    embeddings[image_name] = [batch_embeddings[idx], class_name]
                except Exception as e:
                    print(f"Error with {image_name}: {e}")
                    pass

            if len(embeddings) > 400000:
                # Save the embeddings dictionary to a file
                outfile = output_file.replace(".torch", f"_{counter}.torch")
                torch.save(embeddings, outfile)
                print(f"Saved embeddings dictionary to {outfile}")
                embeddings = {}
                counter += 1

        # after end of for loop, save results        
        outfile = output_file.replace(".torch", f"_{counter}.torch")
        torch.save(embeddings, outfile)
        print(f"Saved embeddings dictionary to {outfile}")