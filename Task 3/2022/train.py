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

test_data = Task3ImageDataset(
                    annotations_file="test_triplets.txt", 
                    img_dir="/home/hjlee/ETH_Spring_2022/IML/Task3/food",
                    transform=transform)

train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True, num_workers=8)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=1)

## Hyperparameters & Log Directory Definition ##
num_epochs = 1000

log_dir_path = os.path.join("/home/hjlee/ETH_Spring_2022/IML/Task3/models", datetime.now().strftime("%m%d_%H%M%S"))
if not os.path.exists(log_dir_path):
    os.makedirs(log_dir_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

## Model Definition ##
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

## Training ##
for epoch in range(num_epochs):
    running_loss = 0.0
    
    for i, (images, labels) in tqdm(enumerate(train_dataloader)):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i%500 == 499:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
            running_loss = 0.0

    ## Saving the model
    if epoch % 10 == 9:
        print("Saving the model... epoch: ", epoch)

        log_path = os.path.join(log_dir_path, f'model_{epoch+1}.pt')
        torch.save(model.state_dict(), log_path)

print('Finished Training')


