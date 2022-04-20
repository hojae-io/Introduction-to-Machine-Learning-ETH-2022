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
from datetime import datetime
from tqdm import tqdm
import time

## Hyperparameters & Log Directory Definition ##
num_epochs = 300
batch_size = 32

log_dir_path = os.path.join("/home/hjlee/ETH_Spring_2022/IML/Task3/models", datetime.now().strftime("%m%d_%H%M%S"))
if not os.path.exists(log_dir_path):
    os.makedirs(log_dir_path)

print('log directory:', log_dir_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

## Dataloader Definition ##
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.ToTensor(),
     transforms.Resize((256,256)),
     normalize, # TODO add resize
     ])

training_data = Task3ImageDataset(
                    annotations_file="train_triplets.txt", 
                    img_dir="/home/hjlee/ETH_Spring_2022/IML/Task3/food",
                    transform=transform)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=8)


## Model Definition ##
model = models.resnet34(pretrained=True)

model.fc = nn.Linear(512, 128)
model.to(device)

# criterion = nn.CrossEntropyLoss()
criterion = nn.TripletMarginLoss(margin=5.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

## Training ##
for epoch in range(num_epochs):
    running_loss = 0.0
    accuracy = 0
    for i, (images, _) in tqdm(enumerate(train_dataloader)):
        img_anchor, img_pos, img_neg = images
        img_anchor, img_pos, img_neg = img_anchor.to(device), img_pos.to(device), img_neg.to(device)

        optimizer.zero_grad()
        embed_anchor = model(img_anchor)

        embed_pos = model(img_pos)
        embed_neg = model(img_neg)

        loss = criterion(embed_anchor, embed_pos, embed_neg)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        pos_norm = torch.norm(embed_anchor - embed_pos, dim=1)
        neg_norm = torch.norm(embed_anchor - embed_neg, dim=1)

        accuracy += torch.sum((pos_norm - neg_norm)<0).item()

        if i%500 == 499:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f} ', end='')
            print(f'accuracy: {accuracy / (500*batch_size):.3f}')
            running_loss = 0.0
            accuracy = 0

    ## Saving the model
    if epoch % 5 == 4:
        print("Saving the model... epoch: ", epoch+1)

        log_path = os.path.join(log_dir_path, f'model_{epoch+1}.pt')
        torch.save(model.state_dict(), log_path)

print('Finished Training')


