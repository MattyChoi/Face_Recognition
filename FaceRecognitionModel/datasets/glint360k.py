import os
import sys

# Import modules from base directory
sys.path.insert(0, os.getcwd())

import json
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class Glint360k(Dataset):
    """
    Dataset for the Glint360k Dataset
    """

    def __init__(
        self,
        data_path="/Users/matthewchoi/Projects/FaceDetection/FaceRecognitionModel/data/img_paths.json",
        transform=None,
        num_sample_per_idty=4,
    ):
        super(Glint360k, self).__init__()

        self.transform = transform
        self.num_sample_per_idty = num_sample_per_idty
        
        assert os.path.exists(data_path), "File does not exist"
            
        with open(data_path, 'r') as file:
            json_dict = json.load(file)
            self.img_paths = list(json_dict["img_paths"].items())
            

    def __getitem__(self, index):
        idty, img_paths = self.img_paths[index]
        idty = int(idty[3:])
        
        # Shuffle img_paths for randomness
        random.shuffle(img_paths)
        
        imgs = []
        num_sample = min(self.num_sample_per_idty, len(img_paths))
        for img_path in img_paths[:num_sample]:
            img = Image.open(img_path).convert("RGB")
            
            # Define a transform to convert the image to a tensor
            transform = transforms.Compose([
                transforms.ToTensor()  # Converts the image to a PyTorch tensor and normalizes to [0, 1]
            ])

            # Apply the transform to the image
            img = transform(img)
            
            imgs.append(img)
            
        return torch.stack(imgs, dim=0), torch.tensor([idty] * num_sample, dtype=torch.long)


    def __len__(self):
        return len(self.img_paths)


    def collate_fn(self, batch):
        imgs, idtys = list(zip(*batch))

        imgs = torch.cat(imgs, dim=0)
        idtys = torch.cat(idtys, dim=0)

        return imgs, idtys


# if __name__ == "__main__":
#     testDataset = Glint360k()
#     first = testDataset[0]
    
#     dl = DataLoader(testDataset, batch_size=4, collate_fn=testDataset.collate_fn)   
#     first_batch = next(iter(dl))