"""
===============================================================================
Title:           dataloader.py
Author:          Vineet Bhat
Date:            June 22, 2024
Description:     This script contains the VGDataLoader() class for loading the RoboRefIt and OCID-VLG dataset. We process and save files as simple .json files for train and test splits. Each entry of the .json file contains a tuple of                 (RGB image path, Mask Path, Referring Text). This script cannot be used standalone. 
===============================================================================
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image,ImageOps
import torchvision.transforms as transforms

transform = transforms.Compose([
            transforms.Resize((352, 352)),
            transforms.ToTensor(),
        ])

class VGDataLoader(Dataset):
    def __init__(self, json_file, invert_mask=False,image_size=352):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        with open(json_file) as f:
            self.data = json.load(f)

        self.image_size = image_size

        self.basic_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

        self.transform = transform

        self.invert_mask = invert_mask


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # Define transformations

        item = self.data[idx]
        master_dir = "./"
        image = Image.open(master_dir+"/"+item['rgb_path'].replace("\\","/")).convert('RGB')
        mask = Image.open(master_dir+"/"+item['mask_path'].replace("\\","/")).convert('L')  # Assuming masks are in grayscale

        if(self.invert_mask):
            mask = ImageOps.invert(mask)

        image = self.basic_transforms(image)
        mask = self.basic_transforms(mask)
        text = item['text']
    
        # Creating an empty tensor for data_x[2]
        empty_tensor = torch.zeros(3, image.shape[1], image.shape[2])

        # data_x and data_y as specified
        data_x = [image, text, empty_tensor, False]
        data_y = [mask, torch.Size([0]), 0]  # Adding an extra dimension to mask to match [batch_size,1,image_width,image_height]

        return data_x, data_y


def get_data_loader(json_file, batch_size=8,invert_mask=False,image_size=352):
    """
    Returns a DataLoader for the VL_Grasp dataset.
    
    Args:
        json_file (string): Path to the json file with annotations.
        batch_size (int): Batch size.
        transform (callable, optional): Optional transform to be applied on a sample.
    
    Returns:
        DataLoader
    """
    dataset = VGDataLoader(json_file,invert_mask,image_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

