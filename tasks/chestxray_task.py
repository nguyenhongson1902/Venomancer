import random
import os

import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch.utils.data import Dataset
import torch.utils.data as torch_data
from torchvision.datasets import ImageFolder
import torchvision
from torchvision.transforms import transforms
from torchvision import models

from transformers import ResNetForImageClassification
# from models.simple import SimpleNet, NetC_MNIST
from models.resnet_chestxray import ResNet18, ResNet50
from tasks.task import Task

from PIL import Image


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
        # img = Image.open(img_path).convert('RGB')  # Convert to RGB
        if self.transform:
            img = self.transform(img)
        return img, label
    

class ChestXRayTask(Task):
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    def load_data(self):
        self.load_chestxray_data()
        number_of_samples = []
        
        if self.params.fl_sample_dirichlet:
            # sample indices for participants using Dirichlet distribution
            # split = min(self.params.fl_total_participants / 20, 1)
            valid = False
            while not valid:
                split = 1.0
                all_range = list(range(int(len(self.train_dataset) * split)))
                self.train_dataset = Subset(self.train_dataset, all_range)
                # indices_per_participant = self.sample_dirichlet_train_data(
                #     self.params.fl_total_participants,
                #     alpha=self.params.fl_dirichlet_alpha)
                indices_per_participant = self.sample_dirichlet_train_data_new(
                    self.params.fl_total_participants,
                    alpha=self.params.fl_dirichlet_alpha)
                # print("DEBUG: ", [len(indices) for indices in indices_per_participant.values()])
                # train_loaders = [self.get_train(indices) for pos, indices in
                #                  indices_per_participant.items()]
                train_loaders, number_of_samples = zip(*[self.get_train(indices) for pos, indices in
                                indices_per_participant.items()])
                
                valid = 0 not in number_of_samples
                if not valid:
                    print("Not valid")
                else:
                    print("Valid")
        else:
            # sample indices for participants that are equally
            # split = min(self.params.fl_total_participants / 20, 1)
            valid = False
            while not valid:
                split = 1.0
                all_range = list(range(int(len(self.train_dataset) * split)))
                self.train_dataset = Subset(self.train_dataset, all_range)
                random.shuffle(all_range)
                # train_loaders = [self.get_train_old(all_range, pos)
                #                  for pos in
                #                  range(self.params.fl_total_participants)]
                train_loaders, number_of_samples = zip(*[self.get_train_old(all_range, pos)
                                                    for pos in range(self.params.fl_total_participants)])
                valid = 0 not in number_of_samples # If there is any client with 0 data, then resample
        self.fl_train_loaders = train_loaders
        self.fl_number_of_samples = number_of_samples
        return

    def load_chestxray_data(self):
        # transform_train = transforms.Compose([
        #     transforms.ToTensor(),
        #     self.normalize
        # ])

        # transform_test = transforms.Compose([
        #     transforms.ToTensor(),
        #     self.normalize
        # ])

        # images_folder = "./.data/dataset/" # accompanied with ImageFolder
        # transform_train = transforms.Compose([
        #     transforms.ToTensor(),
        # ])

        # transform_test = transforms.Compose([
        #     transforms.ToTensor(),
        # ])

        # transform_train = transforms.Compose([
        #     transforms.Resize((256, 256)),
        #     transforms.Grayscale(num_output_channels=1),
        #     transforms.ToTensor(),
        # ]) # ImageFolder

        # transform_test = transforms.Compose([
        #     transforms.Resize((256, 256)),
        #     transforms.Grayscale(num_output_channels=1),
        #     transforms.ToTensor(),
        # ]) # ImageFolder
        

        # self.train_dataset = torchvision.datasets.MNIST(
        #     root=self.params.data_path,
        #     train=True,
        #     download=True,
        #     transform=transform_train)
        # self.train_dataset = ImageFolder(root=os.path.join(images_folder, "train_images_11257"), transform=transform_train)
        
        # train_path = "./.data/dataset/train"
        # test_path = "./.data/dataset/test"
        train_path = "./.data/dataset/train_full"
        test_path = "./.data/dataset/test_full"
        transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.Grayscale(num_output_channels=3),
            # transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.Grayscale(num_output_channels=3),
            # transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        
        self.train_dataset = ChestXRayDataset(root_folder=train_path, transform=transform_train)
        self.train_loader = torch_data.DataLoader(self.train_dataset, batch_size=self.params.batch_size, shuffle=True, num_workers=0)

        # self.train_loader = torch_data.DataLoader(self.train_dataset,
        #                                           batch_size=self.params.batch_size,
        #                                           shuffle=True,
        #                                           num_workers=0)
        # self.test_dataset = torchvision.datasets.MNIST(
        #     root=self.params.data_path,
        #     train=False,
        #     download=True,
        #     transform=transform_test)
        # self.test_dataset = ImageFolder(root=os.path.join(images_folder, "test_images_2252"), transform=transform_test)
        
        self.test_dataset = ChestXRayDataset(root_folder=test_path, transform=transform_test)
        self.test_loader = torch_data.DataLoader(self.test_dataset, batch_size=self.params.test_batch_size, shuffle=False, num_workers=0)

        # self.test_loader = torch_data.DataLoader(self.test_dataset,
        #                                          batch_size=self.params.test_batch_size,
        #                                          shuffle=False,
        #                                          num_workers=0)
        self.classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

        self.clip_image = lambda x: torch.clamp(x, 0.0, 1.0)
        self.target_transform = lambda x: torch.ones_like(x) * self.params.backdoor_label
        return True

    def build_model(self):
        model = ResNet18()
        # print("Loading pretrained ResNet50 model")
        # model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
        # for params in model.parameters():
        #     params.requires_grad = True
        # num_ftrs = model.fc.in_features
        # model.fc = torch.nn.Linear(num_ftrs, self.params.num_classes)

        # model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        # for params in model.parameters():
        #     params.requires_grad = False
        # num_ftrs = model.fc.in_features
        # model.fc = torch.nn.Linear(num_ftrs, self.params.num_classes)

        # model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        # num_ftrs = model.classifier[1].in_features
        # model.classifier[1] = torch.nn.Linear(num_ftrs, self.params.num_classes)



        return model
    