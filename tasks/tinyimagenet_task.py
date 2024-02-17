import random

import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.data import Subset
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from models.resnet_tinyimagenet import ResNet18
from models.resnet_tinyimagenet_dba import resnet18 as resnet18_dba
from torchvision import models
from tasks.task import Task

import os

from utils.backdoor import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_MIN, IMAGENET_MAX
from PIL import Image
import pandas as pd


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir (string): Directory with 'train' and 'val' folders.
            mode (string): 'train' or 'val' to specify which dataset to load.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        assert mode in ['train', 'val'], "Mode must be either 'train' or 'val'"

        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.classes = sorted(os.listdir(os.path.join(root_dir, "train")))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self.get_images()

    def get_images(self):
        images = []
        if self.mode == "train":
            data_dir = os.path.join(self.root_dir, 'train')
            for dir_name in os.listdir(data_dir):
                images_path = os.path.join(data_dir, dir_name, "images")
                for filename in os.listdir(images_path):
                    file_path = os.path.join(images_path, filename)
                    images.append((file_path, self.class_to_idx[dir_name]))
        elif self.mode == "val":
            annotations_file = os.path.join(self.root_dir, "val", "val_annotations.txt")
            annotations = pd.read_csv(annotations_file, sep='\t', header=None)
            annotations.columns = ['filename', 'class_name', '_', '_', '_', '_']
            for i in range(len(annotations)):
                filename = annotations.iloc[i]['filename']
                class_name = annotations.iloc[i]['class_name']
                file_path = os.path.join(self.root_dir, "val", "images", filename)
                images.append((file_path, self.class_to_idx[class_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class TinyImageNetTask(Task):

    def load_data(self):
        self.load_tinyimagenet_data()
        if self.params.fl_sample_dirichlet:
            # sample indices for participants using Dirichlet distribution
            # split = min(self.params.fl_total_participants / 100, 1)
            split = 1.0
            all_range = list(range(int(len(self.train_dataset) * split)))
            self.train_dataset = Subset(self.train_dataset, all_range)
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params.fl_total_participants,
                alpha=self.params.fl_dirichlet_alpha)
            # train_loaders = [self.get_train(indices) for pos, indices in
            #                  indices_per_participant.items()]
            train_loaders, number_of_samples = zip(*[self.get_train(indices) for pos, indices in
                                                    indices_per_participant.items()])
        else:
            # sample indices for participants that are equally
            # split to 500 images per participant
            # split = min(self.params.fl_total_participants / 100, 1)
            split = 1.0
            all_range = list(range(int(len(self.train_dataset) * split)))
            self.train_dataset = Subset(self.train_dataset, all_range)
            random.shuffle(all_range)
            # train_loaders = [self.get_train_old(all_range, pos)
            #                  for pos in
            #                  range(self.params.fl_total_participants)]
            train_loaders, number_of_samples = zip(*[self.get_train_old(all_range, pos) for pos in
                                range(self.params.fl_total_participants)])
        self.fl_train_loaders = train_loaders
        self.fl_number_of_samples = number_of_samples
        return

    def load_tinyimagenet_data(self):

        train_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
        ])

        # dset = {x : torchvision.datasets.ImageFolder(path+x, transform=transformers[y]) for x,y in zip(categories, trans)}
        root_dir = "./.data/tiny-imagenet-200"
        # self.train_dataset = TinyImageNetDataset(root_dir=root_dir, mode="train", transform=train_transform)
        # self.test_dataset = TinyImageNetDataset(root_dir=root_dir, mode="val", transform=test_transform)

        self.train_dataset = ImageFolder(root=os.path.join(root_dir, "train"), transform=train_transform)
        self.test_dataset = ImageFolder(root=os.path.join(root_dir, "val"), transform=test_transform)



        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.params.batch_size,
                                       shuffle=True, num_workers=0)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size,
                                      shuffle=False, num_workers=0)

        self.classes = tuple(range(200))

        self.clip_image = lambda x: torch.clamp(x, 0.0, 1.0)
        self.target_transform = lambda x: torch.ones_like(x) * self.params.backdoor_label
        return True
    
    def build_model(self):
        # model = ResNet18() # from scratch

        # model = ResNet18().to(self.params.device)
        # path = "/hdd/home/ssd_data/Son/Venomancer/saved_models/model_TinyImageNet_02.14_22.06.32_tinyimagenet/model_epoch_25.pt.tar"
        # with open(path, "rb") as f:
        #     checkpoint = torch.load(f, map_location=self.params.device)
        #     model.load_state_dict(checkpoint["state_dict"])
        #     print("Successfully loaded pretrained weights from epoch 25 for ResNet18")
        
        # model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        # for params in model.parameters():
        #     params.requires_grad = True
        # num_ftrs = model.fc.in_features
        # model.fc = torch.nn.Linear(num_ftrs, self.params.num_classes)
        # print("Successfully loaded pretrained weights PyTorch (ImageNet)")

        # model = models.resnet18(weights=None)
        # for params in model.parameters():
        #     params.requires_grad = True
        # num_ftrs = model.fc.in_features
        # model.fc = torch.nn.Linear(num_ftrs, self.params.num_classes)
        # print("Using resnet18 available in PyTorch, train from scratch")

        model = resnet18_dba().to('cuda')
        path = "./pretrained/tiny-resnet.epoch_20"
        with open(path, "rb") as f:
            checkpoint = torch.load(f, map_location="cuda")
            model.load_state_dict(checkpoint['state_dict'])
        print("Using pretrained weights resnet18 from DBA")

        return model
