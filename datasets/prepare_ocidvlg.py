"""
===============================================================================
Title:           prepare_ocidvlg.py
Date:            June 23, 2024
Description:     This file is used to process the OCID VLG Dataset and convert it to the required format for training and testing HiFi-CS
Usage:           python datasets/prepare_ocidvlg.py
===============================================================================
"""

import os
import numpy as np
from PIL import Image
import json
import sys
from tqdm import tqdm
from load_ocidvlg import OCIDVLGDataset
sys.path.append("./datasets/OCID-VLG/")

# Root directory for extracting the images
root_dir = "./datasets/OCID-VLG/"

def save_image(np_array, path, mode='RGB'):
    """
    Saves a numpy array as an image.
    
    Parameters:
        np_array (numpy.ndarray): The image data in numpy array format.
        path (str): Path where the image will be saved.
        mode (str): The mode to convert the image to. Default is 'RGB'.
    """

    if isinstance(np_array, Image.Image):
        image = np.array(np_array)


    # Normalize if the image is depth or mask and in floating point format
    if mode != 'RGB':  # Assuming the need for normalization for non-RGB (e.g., depth and mask images)
        # Normalize if the image is depth or mask and potentially in floating point format
        image = image.astype(float)  # Ensure array is in float format for normalization
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype('uint8')

    image = Image.fromarray(image)
    if image.mode != mode:
        image = image.convert(mode)
    
    # Save the image
    image.save(path)

# Function to ensure directories exist
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
        
print("Processing Train set")
base_dir = "./datasets/ocidvlg_final_dataset/train"

# Specific directories for images, depth, and masks
img_dir = os.path.join(base_dir, "image")
depth_dir = os.path.join(base_dir, "depth")
mask_dir = os.path.join(base_dir, "mask")

# Ensure the directories exist
ensure_dir(img_dir)
ensure_dir(depth_dir)
ensure_dir(mask_dir)

dataset_ = OCIDVLGDataset(root_dir, split="train", version="unique")

data_entries = []

print("Total data to process = "+str(len(dataset_)))

for i in tqdm(range(len(dataset_)), desc="Processing"):
    sample=dataset_[i]
    # Process and save RGB image
    img_path = os.path.join(img_dir, f"{i:07d}.png").replace("./datasets/","./")
    img = Image.fromarray(sample["img"]).resize((352, 352))
    img.save(img_path)

    # Process and save depth image
    depth_path = os.path.join(depth_dir, f"{i:07d}.png").replace("./datasets/","./")
    depth = Image.fromarray(sample["depth"]).resize((352, 352))
    save_image(depth, depth_path, mode='L')

    # Process and save mask
    mask_individual_dir = os.path.join(mask_dir, f"{i:07d}")
    #ensure_dir(mask_individual_dir)  # Ensure each mask's directory exists
    mask_path = os.path.join(mask_dir, f"{i:07d}.png").replace("./datasets/","./")  # Assuming a single mask per image for this example
    mask = Image.fromarray(sample["mask"]).resize((352, 352))
    save_image(mask, mask_path, mode='L')

    # Create and append the data entry
    data_entry = {
        "num": i,
        "text": sample["sentence"],
        "rgb_path": img_path.replace("\\", "/"),  # Ensure consistent path format
        "depth_path": depth_path.replace("\\", "/"),
        "mask_path": mask_path.replace("\\", "/"),
    }
    data_entries.append(data_entry)


# Save the JSON file
json_path = os.path.join(base_dir, "ocid_vlg_train.json")
with open(json_path, 'w') as f:
    json.dump(data_entries, f, indent=4)
    
    
print("Processing Test set")
base_dir = "./datasets/ocidvlg_final_dataset/test"

# Specific directories for images, depth, and masks
img_dir = os.path.join(base_dir, "image")
depth_dir = os.path.join(base_dir, "depth")
mask_dir = os.path.join(base_dir, "mask")

# Ensure the directories exist
ensure_dir(img_dir)
ensure_dir(depth_dir)
ensure_dir(mask_dir)

dataset_ = OCIDVLGDataset(root_dir, split="test", version="unique")

data_entries = []

print("Total data to process = "+str(len(dataset_)))

for i in tqdm(range(len(dataset_)), desc="Processing"):
    sample=dataset_[i]
    # Process and save RGB image
    img_path = os.path.join(img_dir, f"{i:07d}.png").replace("./datasets/","./")
    img = Image.fromarray(sample["img"]).resize((352, 352))
    img.save(img_path)

    # Process and save depth image
    depth_path = os.path.join(depth_dir, f"{i:07d}.png").replace("./datasets/","./")
    depth = Image.fromarray(sample["depth"]).resize((352, 352))
    save_image(depth, depth_path, mode='L')

    # Process and save mask
    mask_individual_dir = os.path.join(mask_dir, f"{i:07d}")
    #ensure_dir(mask_individual_dir)  # Ensure each mask's directory exists
    mask_path = os.path.join(mask_dir, f"{i:07d}.png").replace("./datasets/","./")  # Assuming a single mask per image for this example
    mask = Image.fromarray(sample["mask"]).resize((352, 352))
    save_image(mask, mask_path, mode='L')

    # Create and append the data entry
    data_entry = {
        "num": i,
        "text": sample["sentence"],
        "rgb_path": img_path.replace("\\", "/"),  # Ensure consistent path format
        "depth_path": depth_path.replace("\\", "/"),
        "mask_path": mask_path.replace("\\", "/"),
    }
    data_entries.append(data_entry)


# Save the JSON file
json_path = os.path.join(base_dir, "ocid_vlg_test.json")
with open(json_path, 'w') as f:
    json.dump(data_entries, f, indent=4)
