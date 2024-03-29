import random

import torch
from torch.utils.data import Subset
import torch.utils.data as torch_data
import torchvision
from torchvision.transforms import transforms

# from models.simple import SimpleNet, NetC_MNIST
from models.resnet_mnist import ResNet18
from models import vgg9_only as vgg9
from models import vgg
from tasks.task import Task


class MNISTTask(Task):
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    def load_data(self):
        self.load_mnist_data()
        number_of_samples = []
        
        if self.params.fl_sample_dirichlet:
            # sample indices for participants using Dirichlet distribution
            # split = min(self.params.fl_total_participants / 20, 1)
            split = 1.0
            all_range = list(range(int(len(self.train_dataset) * split)))
            self.train_dataset = Subset(self.train_dataset, all_range)
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params.fl_total_participants,
                alpha=self.params.fl_dirichlet_alpha)
            # print("DEBUG: ", [len(indices) for indices in indices_per_participant.values()])
            # train_loaders = [self.get_train(indices) for pos, indices in
            #                  indices_per_participant.items()]
            train_loaders, number_of_samples = zip(*[self.get_train(indices) for pos, indices in
                             indices_per_participant.items()])
        else:
            # sample indices for participants that are equally
            # split = min(self.params.fl_total_participants / 20, 1)
            split = 1.0
            all_range = list(range(int(len(self.train_dataset) * split)))
            self.train_dataset = Subset(self.train_dataset, all_range)
            random.shuffle(all_range)
            # train_loaders = [self.get_train_old(all_range, pos)
            #                  for pos in
            #                  range(self.params.fl_total_participants)]
            train_loaders, number_of_samples = zip(*[self.get_train_old(all_range, pos)
                                                for pos in range(self.params.fl_total_participants)])
        self.fl_train_loaders = train_loaders
        self.fl_number_of_samples = number_of_samples
        return

    def load_mnist_data(self):
        # transform_train = transforms.Compose([
        #     transforms.ToTensor(),
        #     self.normalize
        # ])

        # transform_test = transforms.Compose([
        #     transforms.ToTensor(),
        #     self.normalize
        # ])

        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.train_dataset = torchvision.datasets.MNIST(
            root=self.params.data_path,
            train=True,
            download=True,
            transform=transform_train)
        self.train_loader = torch_data.DataLoader(self.train_dataset,
                                                  batch_size=self.params.batch_size,
                                                  shuffle=True,
                                                  num_workers=0)
        self.test_dataset = torchvision.datasets.MNIST(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform_test)
        self.test_loader = torch_data.DataLoader(self.test_dataset,
                                                 batch_size=self.params.test_batch_size,
                                                 shuffle=False,
                                                 num_workers=0)
        self.classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

        self.clip_image = lambda x: torch.clamp(x, 0.0, 1.0)
        self.target_transform = lambda x: torch.ones_like(x) * self.params.backdoor_label
        return True

    def build_model(self):
        model = ResNet18()

        # model = vgg9.VGG('VGG9', in_channels=1, num_classes=10)
        # print("Train VGG9 from scratch")

        # model = vgg.get_vgg_model('vgg11', num_classes=10, task='mnist')
        # print("Train VGG11 from scratch")

        return model
    