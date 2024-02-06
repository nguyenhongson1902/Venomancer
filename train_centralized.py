import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

import wandb
from models.resnet_chestxray import ResNet50
from torchvision import models
from tqdm import tqdm

import sys
import os


class ChestXRayDataset(Dataset):
    def __init__(self, root_folder, transform=None):
        self.root_folder = root_folder
        self.transform = transform
        self.classes = sorted(os.listdir(root_folder))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self.get_images()

    def get_images(self):
        images = []
        for cls in self.classes:
            class_folder = os.path.join(self.root_folder, cls)
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                images.append((img_path, self.class_to_idx[cls]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            img = self.transform(img)
        return img, label

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# Create train and test datasets
train_dataset = ChestXRayDataset(root_folder='./.data/dataset/train_images_11257', transform=transform)
test_dataset = ChestXRayDataset(root_folder='./.data/dataset/test_images_2252', transform=transform)

# Create train and test dataloaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


# wandb.login(key="917b44927c77ee61ea91005724c9bd9b470f116a")
# wandb.init(project="backdoor-attack", entity="nguyenhongsonk62hust", name=f"Test ChestXRay_Centralized", dir="./hdd/home/ssd_data/Son/Venomancer/wandb")

model = models.resnet50(weights="ResNet50_Weights.DEFAULT").to('cuda')


num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 15).to('cuda')

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()

for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
    inputs, labels = data
    inputs, labels = inputs.to('cuda'), labels.to('cuda')
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    
    optimizer.step()
    
    




