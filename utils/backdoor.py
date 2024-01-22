import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import os
import csv
import kornia.augmentation as A
import random
import numpy as np
import copy
from tqdm import tqdm

from PIL import Image
# from torch.utils.tensorboard import SummaryWriter

# def all2one_target_transform(x, attack_target=1):
#     return torch.ones_like(x) * attack_target

# def all2all_target_transform(x, num_classes):
#     return (x + 1) % num_classes

# def get_target_transform(hlpr):
#     """Get target transform function
#     """
#     if hlpr.params.mode == 'all2one':
#         target_transform = lambda x: all2one_target_transform(x, hlpr.params.target_label)
#     elif hlpr.params.mode == 'all2all':
#         target_transform = lambda x: all2all_target_transform(x, hlpr.params.num_classes)
#     else:
#         raise Exception(f'Invalid mode {hlpr.params.mode}')
#     return target_transform

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_MIN  = ((np.array([0,0,0]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).min()
IMAGENET_MAX  = ((np.array([1,1,1]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).max()
class ToNumpy:
    def __call__(self, x):
        x = np.array(x)
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        return x

class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):  # , **kwargs):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x


def get_transform(opt, train=True, pretensor_transform=False):
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    if pretensor_transform:
        if train:
            transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop))
            transforms_list.append(transforms.RandomRotation(opt.random_rotation))
            if opt.dataset == "cifar10":
                transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))

    transforms_list.append(transforms.ToTensor())
    
    if opt.dataset == 'mnist':
        transforms_list.append(transforms.Normalize([0.5], [0.5]))
    else:
        transforms_list.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    
    return transforms.Compose(transforms_list)


class PostTensorTransform(torch.nn.Module):
    def __init__(self, opt):
        super(PostTensorTransform, self).__init__()
        self.random_crop = ProbTransform(
            A.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop), p=0.8
        )
        self.random_rotation = ProbTransform(A.RandomRotation(opt.random_rotation), p=0.5)
        if opt.dataset == "cifar10":
            self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    
class GTSRB(data.Dataset):
    def __init__(self, opt, train, transforms, data_root=None, min_width=0):
        super(GTSRB, self).__init__()
        if data_root is None:
            data_root = opt.data_root
        if train:
            self.data_folder = os.path.join(data_root, "GTSRB/Train")
            self.images, self.labels = self._get_data_train_list(min_width=min_width)
            if min_width > 0:
                print(f'Loading GTSRB Train greater than {min_width} width. Loaded {len(self.images)} images.')
        else:
            self.data_folder = os.path.join(data_root, "GTSRB/Test")
            self.images, self.labels = self._get_data_test_list(min_width)
            print(f'Loading GTSRB Test greater than {min_width} width. Loaded {len(self.images)} images.')

        self.transforms = transforms

    def _get_data_train_list(self, min_width=0):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                if int(row[1]) >= min_width:
                    images.append(prefix + row[0])
                    labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self, min_width=0):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            if int(row[1]) >= min_width: #only load images if more than certain width
                images.append(self.data_folder + "/" + row[0])
                labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label

class CelebA_attr(data.Dataset):
    def __init__(self, opt, split, transforms):
        self.dataset = torchvision.datasets.CelebA(root=opt.data_root, split=split, target_type="attr", download=True)
        self.list_attributes = [18, 31, 21]
        self.transforms = transforms
        self.split = split

    def _convert_attributes(self, bool_attributes):
        return (bool_attributes[0] << 2) + (bool_attributes[1] << 1) + (bool_attributes[2])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        input = self.transforms(input)
        target = self._convert_attributes(target[self.list_attributes])
        return (input, target)

def get_dataloader(opt, train=True, pretensor_transform=False, min_width=0):
    transform = get_transform(opt, train, pretensor_transform)
    if opt.dataset == "gtsrb":
        dataset = GTSRB(opt, train, transform, min_width=min_width)
    elif opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(opt.data_root, train, transform, download=True)
    elif opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True)
    elif opt.dataset == "celeba":
        if train:
            split = "train"
        else:
            split = "test"
        dataset = CelebA_attr(opt, split, transform)
    elif opt.dataset in ['tiny-imagenet', 'tiny-imagenet32']:
        if train:
            split = 'train'
        else:
            split = 'test'
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(opt.data_root, 'tiny-imagenet-200', split), transform=transform)
    else:
        raise Exception("Invalid dataset")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
    return dataloader


def get_dataset(opt, train=True):
    if opt.dataset == "gtsrb":
        dataset = GTSRB(
            opt,
            train,
            transforms=transforms.Compose([transforms.Resize((opt.input_height, opt.input_width)), ToNumpy()]),
        )
    elif opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(opt.data_root, train, transform=ToNumpy(), download=True)
    elif opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform=ToNumpy(), download=True)
    elif opt.dataset == "celeba":
        if train:
            split = "train"
        else:
            split = "test"
        dataset = CelebA_attr(
            opt,
            split,
            transforms=transforms.Compose([transforms.Resize((opt.input_height, opt.input_width)), ToNumpy()]),
        )
    elif opt.dataset in ['tiny-imagenet', 'tiny-imagenet32']:
        if train:
            split = 'train'
        else:
            split = 'test'
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(opt.data_root, 'tiny-imagenet-200', split), 
            transform=transforms.Compose([transforms.Resize((opt.input_height, opt.input_width)), ToNumpy()]))
    else:
        raise Exception("Invalid dataset")
    return dataset


# def get_target_transform(hlpr):
#     return lambda x: torch.ones_like(x) * hlpr.params.target_label

def make_backdoor_batch(hlpr, data, target, atkmodel, target_transform, multitarget=False):
    # print("atkmodel", type(atkmodel))
    if multitarget:
        atktarget = target_transform(target, n_classes=10)
        # noise = atkmodel(data, atktarget) * hlpr.params.eps
        noise = atkmodel(data, atktarget)
        # noise = torch.clamp(noise, -hlpr.params.eps, hlpr.params.eps)
        atkdata = hlpr.task.clip_image(data + noise)
    else:
        # noise = atkmodel(data) * hlpr.params.eps
        noise = atkmodel(data)
        # noise = torch.clamp(noise, -hlpr.params.eps, hlpr.params.eps)
        atkdata = hlpr.task.clip_image(data + noise)
        atktarget = target_transform(target)

    return atkdata, atktarget

def aggregate_atkmodels(hlpr, atkmodels_dict, round_participants):
    # print(atkmodels_dict.keys())

    adversaries = [user for user in round_participants if user.compromised]
    sum_malicious_samples = sum([user.number_of_samples for user in adversaries])
    weight_contribution = {}
    for user in adversaries:
        weight_contribution[user.user_id] = user.number_of_samples / sum_malicious_samples

    # Initialize a new atkmodel
    atkmodel_avg, tgtmodel_avg, _ = hlpr.task.get_atkmodel()
    
    for param in atkmodel_avg.parameters():
        param.data.fill_(0.0)
    
    for param in tgtmodel_avg.parameters():
        param.data.fill_(0.0)
    
    # Sum the weights
    for user_id, backdoor_package in atkmodels_dict.items():
        atkmodel = backdoor_package[0]
        tgtmodel = backdoor_package[1]

        atkmodel_weights = atkmodel.state_dict()
        my_dict = atkmodel_avg.state_dict()
        # Multiply the weights by the weight contribution
        for key in atkmodel_weights.keys():
            if hlpr.attack.check_ignored_weights(key):
                continue
            my_dict[key] += weight_contribution[user_id] * atkmodel_weights[key]
        
        atkmodel_avg.load_state_dict(my_dict) # Double check

        tgtmodel_weights = tgtmodel.state_dict()
        my_dict_tmp = tgtmodel_avg.state_dict()
        # Multiply the weights by the weight contribution
        for key in tgtmodel_weights.keys():
            if hlpr.attack.check_ignored_weights(key):
                continue
            my_dict_tmp[key] += weight_contribution[user_id] * tgtmodel_weights[key]
        
        tgtmodel_avg.load_state_dict(my_dict_tmp) # Double check
    
    tgtoptimizer_avg = torch.optim.Adam(tgtmodel_avg.parameters(), lr=hlpr.params.lr_atk) # Initialize a new tgtoptimizer for the average model

    for user_id in atkmodels_dict.keys():
        atkmodels_dict[user_id] = [atkmodel_avg, tgtmodel_avg, tgtoptimizer_avg]

    return atkmodel_avg, tgtmodel_avg, tgtoptimizer_avg

def pick_best_atkmodel(hlpr, atkmodels_dict, round_participants, malicious_local_models):
    all_accs = []
    adversaries = [user for user in round_participants if user.compromised]
    
    for user in adversaries:
        atkmodel = atkmodels_dict[user.user_id][0]
        tgtmodel = atkmodels_dict[user.user_id][1]
        local_model = malicious_local_models[user.user_id]

        backdoor_loss, backdoor_correct = 0.0, 0
        batch_size = 0
        with torch.no_grad():
            for i, data_labels in enumerate(user.train_loader):
                batch = hlpr.task.get_batch(i, data_labels)
                
                data, target = batch.inputs, batch.labels
                batch_size += data.shape[0]

                # target_transform = hlpr.task.target_transform
                target_transform = hlpr.task.sample_negative_labels
                # atkdata, atktarget = make_backdoor_batch(hlpr, data, target, atkmodel, target_transform, multitarget=False)
                atkdata, atktarget = make_backdoor_batch(hlpr, data, target, atkmodel, target_transform, multitarget=True)
                atkoutput = local_model(atkdata)

                backdoor_loss += hlpr.task.criterion(atkoutput, atktarget).sum().item()
                atkpred = atkoutput.max(1, keepdim=True)[1]  # get the index of the max log-probability
                backdoor_correct += atkpred.eq(atktarget.view_as(atkpred)).sum().item()

        backdoor_loss /= batch_size
        backdoor_acc = backdoor_correct / batch_size

        all_accs.append(backdoor_acc)
    
    arg_idx = np.argmax(np.array(all_accs))
    best_malicious_user = adversaries[arg_idx]

    best_atkmodel = atkmodels_dict[best_malicious_user.user_id][0]
    best_tgtmodel = atkmodels_dict[best_malicious_user.user_id][1]
    
    best_tgtoptimizer = torch.optim.Adam(best_tgtmodel.parameters(), lr=hlpr.params.lr_atk, betas=(0.5, 0.999)) # Initialize a new tgtoptimizer for the average model
    atkmodels_dict[best_malicious_user.user_id][2] = best_tgtoptimizer

    local_backdoor_acc = all_accs[arg_idx]

    return best_atkmodel, best_tgtmodel, best_tgtoptimizer, local_backdoor_acc

def pick_backdoor_label_samples(hlpr, data, target):
    backdoor_label = hlpr.params.backdoor_label
    # Get indices where target is equal to 1
    indices = (target == backdoor_label).nonzero(as_tuple=True)[0]

    # Use the indices to select data with target==1
    backdoor_label_data = data[indices]
    backdoor_label_target = target[indices]
    return backdoor_label_data, backdoor_label_target

# Used for durability (Neurotoxin)
def apply_grad_mask(model, mask_grad_list):
    mask_grad_list_copy = iter(mask_grad_list)
    for name, parms in model.named_parameters():
        if parms.requires_grad:
            parms.grad = parms.grad * next(mask_grad_list_copy)

def get_grad_mask(hlpr, local_model, local_optimizer, clean_dataloader, history_grad_list_neurotoxin, ratio=0.95):
    """
    Generate a gradient mask based on the given dataset
    This function is employed for Neurotoxin method
    https://proceedings.mlr.press/v162/zhang22w.html
    """        

    local_model.train()
    local_model.zero_grad()
    # loss_fn = nn.CrossEntropyLoss()
    # Let's assume we have a model trained on clean data and we conduct aggregation for all layer
    for batch_idx, data_labels in enumerate(clean_dataloader):
        batch = hlpr.task.get_batch(batch_idx, data_labels)
        bs = batch.batch_size
        data, targets = batch.inputs, batch.labels

        clean_images, clean_targets = copy.deepcopy(data).to(hlpr.params.device), copy.deepcopy(targets).to(hlpr.params.device)
        # clean_images, clean_targets = data, targets
        local_optimizer.zero_grad()
        output = local_model(clean_images)
        loss_clean = hlpr.task.criterion(output, clean_targets)
        # loss_clean.backward(retain_graph=True)
        loss_clean.mean().backward()


    mask_grad_list = []
    grad_list = []
    grad_abs_sum_list = []
    k_layer = 0
    for _, params in local_model.named_parameters():
        if params.requires_grad:
            grad_list.append(params.grad.abs().view(-1))
            grad_abs_sum_list.append(params.grad.abs().view(-1).sum().item())
            k_layer += 1

    grad_list = torch.cat(grad_list).to(hlpr.params.device)
    if len(history_grad_list_neurotoxin) >= 10:
        history_grad_list_neurotoxin.pop(0) # Prevent memory to overflow
    history_grad_list_neurotoxin.append(grad_list)
    grad_list = sum(history_grad_list_neurotoxin) / len(history_grad_list_neurotoxin)
    # history_grad_list_neurotoxin.append(grad_list)
    # if len(history_grad_list_neurotoxin) == 2:
    #     grad_list = 0.9*history_grad_list_neurotoxin[0] + 0.1*history_grad_list_neurotoxin[1]
    
    # grad_list = grad_list.to(hlpr.params.device)

    _, indices = torch.topk(-1*grad_list, int(len(grad_list)*ratio))
    mask_flat_all_layer = torch.zeros(len(grad_list)).to(hlpr.params.device)
    mask_flat_all_layer[indices] = 1.0
    

    count = 0
    percentage_mask_list = []
    k_layer = 0
    grad_abs_percentage_list = []
    for _, parms in local_model.named_parameters():
        if parms.requires_grad:
            gradients_length = len(parms.grad.abs().view(-1))

            mask_flat = mask_flat_all_layer[count:count + gradients_length ].to(hlpr.params.device)
            mask_grad_list.append(mask_flat.reshape(parms.grad.size()).to(hlpr.params.device))

            count += gradients_length

            percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0

            percentage_mask_list.append(percentage_mask1)

            grad_abs_percentage_list.append(grad_abs_sum_list[k_layer]/np.sum(grad_abs_sum_list))

            k_layer += 1
    return mask_grad_list
