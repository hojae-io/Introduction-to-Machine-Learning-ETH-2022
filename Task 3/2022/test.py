import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from dataloader import Task3ImageDataset
from tqdm import tqdm
"""
Testing (Implementation on "test_triplets.txt")
"""

## Dataloader Definition ##
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.ToTensor(),
     transforms.Resize((256,256)),
     normalize, # TODO add resize
     ])


test_data = Task3ImageDataset(
                    annotations_file="test_triplets.txt", 
                    img_dir="/home/hjlee/ETH_Spring_2022/IML/Task3/food",
                    transform=transform)

test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)
## Model Definition ##
model = models.resnet34(pretrained=True)
model.fc = nn.Linear(512, 128)

model_path = "/home/hjlee/ETH_Spring_2022/IML/Task3/models/0420_132004/model_15.pt"
model.load_state_dict(torch.load(model_path, map_location="cuda:0"))
model.to(device)

model.eval()
## Testing ##
with open('result.txt', 'w') as f:
    with torch.inference_mode():
        for i, (images, _) in tqdm(enumerate(test_dataloader)):
            img_anchor, img_q1, img_q2 = images
            img_anchor, img_q1, img_q2 = img_anchor.to(device), img_q1.to(device), img_q2.to(device)
            
            embed_anchor = model(img_anchor)
            embed_q1 = model(img_q1)
            embed_q2 = model(img_q2)

            q1_norm = torch.norm(embed_anchor - embed_q1, dim=1)
            q2_norm = torch.norm(embed_anchor - embed_q2, dim=1)

            if q1_norm < q2_norm:
                result = "1"
            else:
                result = "0"

            f.write(result)
            f.write('\n')



