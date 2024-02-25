import random
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import transforms

# from models.resnet import resnet18
from models.resnet_cifar100_v2 import ResNet18 as ResNet18_v2
from models.resnet_cifar100_v1 import ResNet18 as ResNet18_v1
import models.resnet_cifar10_resnet20 as resnet
from models.resnet_pytorch_cifar100 import resnet18
from tasks.task import Task

from utils.backdoor import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_MIN, IMAGENET_MAX
import torch
# from utils.dataloader import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class CIFAR100Task(Task):
    # normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                                  (0.2023, 0.1994, 0.2010))
    normalize = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    # resize = transforms.Resize((32, 32)) # Marksman
    

    def load_data(self):
        self.load_cifar_data()
        number_of_samples = []

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
            # train_loaders = [self.get_iid_train_loader(all_range, pos)
            #                  for pos in
            #                  range(self.params.fl_total_participants)]

            train_loaders, number_of_samples = zip(*[self.get_train_old(all_range, pos) for pos in
                                                     range(self.params.fl_total_participants)])
        self.fl_train_loaders = train_loaders
        self.fl_number_of_samples = number_of_samples
        return

    def load_cifar_data(self):
        # if self.params.transform_train:
        #     transform_train = transforms.Compose([
        #         transforms.RandomCrop(32, padding=4),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         self.normalize,
        #     ])
        # else:
        #     transform_train = transforms.Compose([
        #         transforms.ToTensor(),
        #         self.normalize,
        #     ])
        
        # transform_train = transforms.Compose([transforms.ToTensor(), self.normalize,])
        transform_train = transforms.Compose([transforms.ToTensor(), ]) # My method
        
        # A3FL transformations
        # transform_train = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # ])

        
        # transform_test = transforms.Compose([transforms.ToTensor(), self.normalize])
        transform_test = transforms.Compose([transforms.ToTensor(), ]) # My method

        # A3FL transformations
        # transform_test = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # ])

        self.train_dataset = torchvision.datasets.CIFAR100(
            root=self.params.data_path,
            train=True,
            download=True,
            transform=transform_train)

        self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=self.params.batch_size,
                                           shuffle=True,
                                           num_workers=0)
        self.test_dataset = torchvision.datasets.CIFAR100(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform_test)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size,
                                      shuffle=False, num_workers=0)

        self.classes = tuple(range(100))
        
        # self.clip_image = lambda x: torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
        self.clip_image = lambda x: torch.clamp(x, 0.0, 1.0)
        self.target_transform = lambda x: torch.ones_like(x) * self.params.backdoor_label
        return True

    def build_model(self) -> nn.Module:
        # model = resnet18(pretrained=False,
        #                 num_classes=len(self.classes))
        # from torchvision.models import resnet18, ResNet18_Weights
        # weights = ResNet18_Weights.DEFAULT
        # model = resnet18(weights=weights)
        # num_features = model.fc.in_features
        # model.fc = nn.Linear(num_features, 10)
        
        # model = ResNet18().to("cuda")
        # with open("/home/vishc2/sonnh/Venomancer/pretrained/model_last.pt.tar.epoch_200", "rb") as f:
        #     checkpoint = torch.load(f, map_location="cuda")
        #     model.load_state_dict(checkpoint["state_dict"])

        # model = ResNet18()
        # path = "./pretrained/cifar100-resnet18.epoch_10"
        # with open(path, "rb") as f:
        #     checkpoint = torch.load(f)
        #     model.load_state_dict(checkpoint, strict=False)
        # print("Loaded cifar100-resnet18.epoch_10")

        # model = ResNet18_dba().to('cuda')
        # path = "./pretrained/model_last.pt.tar.epoch_200"
        # with open(path, "rb") as f:
        #     checkpoint = torch.load(f, map_location="cuda")
        #     model.load_state_dict(checkpoint['state_dict'])
        # print("Loaded pretrained weights resnet18 DBA")

        # model = getattr(resnet, "cifar100_resnet20")()
        # checkpoint = torch.load('./pretrained/cifar100_resnet20-23dac2f1.pt')
        # model.load_state_dict(checkpoint)
        # print("Use pretrained weights resnet20 (github https://github.com/chenyaofo/pytorch-cifar-models/tree/logs)")

        model = ResNet18_v1()
        print("Training ResNet18_v1 from scratch")

        # model = resnet18()
        # checkpoint = torch.load("./pretrained/resnet18-200-cifar100.pth")
        # model.load_state_dict(checkpoint)
        # print("Use pretrained weights resnet20 (github https://github.com/weiaicunzai/pytorch-cifar100)")

        
        return model