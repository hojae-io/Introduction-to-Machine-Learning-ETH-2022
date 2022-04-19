import numpy as np
import os
import pandas as pd
from torchvision.io import read_image

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import random

class Task3ImageDataset(Dataset):
    """
    annotation_file: train_triplets.txt
    img_dir: /home/hjlee/ETH_Spring_2022/IML/Task3/food
    """
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        with open(annotations_file) as f:
            self.img_labels = f.readlines() # self.img_labels is list of strings such as '01809 02962 00582\n'
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        """ 
        image: width-wise concatenation of three transformed images
        label: 0 or 1
         """
        img_label = self.img_labels[idx][:-1].split(' ')
        img0_path = os.path.join(self.img_dir, img_label[0]+'.jpg')
        img1_path = os.path.join(self.img_dir, img_label[1]+'.jpg')
        img2_path = os.path.join(self.img_dir, img_label[2]+'.jpg')
        image0 = read_image(img0_path)
        image1 = read_image(img1_path)
        image2 = read_image(img2_path)

        # switch = random.choice([True, False])

        if self.transform:
            image0 = self.transform(image0) # Normalize and resize to (3, 256, 256)
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        
        # if switch:
        #     image = torch.cat((image0, image2, image1), 2) # -> [3, 256, 256x3]
        #     label = 0 # 0: img0 is closer to img2 than to img1
        # else:
        #     image = torch.cat((image0, image1, image2), 2)
        #     label = 1 # 1: img0 is closer to img1 than to img2
        label = 1
        image = [image0, image1, image2] # image0: anchor, image1: positive, image2: negative
        
        return image, label