import logging
import random
from typing import List, Any, Dict
from copy import deepcopy
import numpy as np
from collections import defaultdict

import torch
from torch import optim, nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms

from metrics.accuracy_metric import AccuracyMetric
from metrics.metric import Metric
from metrics.test_loss_metric import TestLossMetric
from tasks.batch import Batch
from tasks.fl_user import FLUser
from utils.parameters import Params

logger = logging.getLogger('logger')


class Task:
    params: Params = None

    train_dataset = None
    test_dataset = None
    train_loader = None
    test_loader = None
    classes = None

    model: Module = None
    optimizer: optim.Optimizer = None
    # scheduler = None
    criterion: Module = None
    metrics: List[Metric] = None

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    "Generic normalization for input data."
    input_shape: torch.Size = None

    fl_train_loaders: List[Any] = None
    fl_number_of_samples: List[int] = None

    ignored_weights = ['num_batches_tracked']#['tracked', 'running']
    adversaries: List[int] = None

    def __init__(self, params: Params):
        self.params = params
        self.init_task()

    def init_task(self):
        self.load_data()
        logger.debug(f"Number of train samples: {self.fl_number_of_samples}")
        self.model = self.build_model()
        self.resume_model()
        self.model = self.model.to(self.params.device) # global model

        self.local_model = self.build_model().to(self.params.device)
        self.criterion = self.make_criterion()
        self.adversaries = self.sample_adversaries()

        self.optimizer = self.make_optimizer()
        self.metrics = [AccuracyMetric(), TestLossMetric(self.criterion)]
        self.set_input_shape()

        # Initialize the logger
        fh = logging.FileHandler(
                filename=f'{self.params.folder_path}/log.txt')
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    def load_data(self) -> None:
        raise NotImplemented

    def build_model(self) -> Module:
        raise NotImplemented

    def make_criterion(self) -> Module:
        """Initialize with Cross Entropy by default.

        We use reduction `none` to support gradient shaping defense.
        :return:
        """
        return nn.CrossEntropyLoss(reduction='none')
        # return nn.CrossEntropyLoss(reduction='mean')

    def make_optimizer(self, model=None) -> Optimizer:
        if model is None:
            model = self.model # global model
        if self.params.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(),
                                  lr=self.params.lr,
                                  weight_decay=self.params.decay,
                                  momentum=self.params.momentum)
            # self.scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0.001)
        elif self.params.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(),
                                   lr=self.params.lr,
                                   weight_decay=self.params.decay)
            # optimizer = optim.Adam(model.parameters(),
            #                        lr=self.params.lr,
            #                        betas=(0.5, 0.999),
            #                        ) # Starts from exp67
        else:
            raise ValueError(f'No optimizer: {self.optimizer}')

        return optimizer

    def resume_model(self):
        if self.params.resume_model:
            logger.info(f'Resuming training from {self.params.resume_model}')
            prefix = self.params.prefix
            loaded_params = torch.load(prefix + f"saved_models/"
                                       f"{self.params.resume_model}",
                                    map_location=torch.device('cpu'))
            self.model.load_state_dict(loaded_params['state_dict'])

            self.params.eps = loaded_params['eps'] # Start from exp75
            # self.params.start_epoch = loaded_params['epoch']
            # self.params.lr = loaded_params.get('lr', self.params.lr)

            # logger.warning(f"Loaded parameters from saved model: LR is"
            #                f" {self.params.lr} and current epoch is"
            #                f" {self.params.start_epoch}")
            logger.warning(f"Loaded parameters from saved model: LR is"
                           f" {self.params.lr} and previous epoch is"
                           f" {loaded_params['epoch']}")

    def set_input_shape(self):
        inp = self.train_dataset[0][0]
        self.params.input_shape = inp.shape
        logger.info(f'Input shape: {self.params.input_shape}')

    def get_batch(self, batch_id, data) -> Batch:
        """Process data into a batch.

        Specific for different datasets and data loaders this method unifies
        the output by returning the object of class Batch.
        :param batch_id: id of the batch
        :param data: object returned by the Loader.
        :return:
        """
        inputs, labels = data
        batch = Batch(batch_id, inputs, labels)
        return batch.to(self.params.device)

    def accumulate_metrics(self, outputs, labels):
        for metric in self.metrics:
            metric.accumulate_on_batch(outputs, labels)

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_metric()

    def report_metrics(self, step, prefix=''):
        metric_text = []
        for metric in self.metrics:
            metric_text.append(str(metric))
        logger.warning(f'{prefix} {step:4d}. {" | ".join(metric_text)}')

        return  self.metrics[0].get_main_metric_value()

    @staticmethod
    def get_batch_accuracy(outputs, labels, top_k=(1,)):
        """Computes the precision@k for the specified values of k"""
        max_k = max(top_k)
        batch_size = labels.size(0)

        _, pred = outputs.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append((correct_k.mul_(100.0 / batch_size)).item())
        if len(res) == 1:
            res = res[0]
        return res

    def get_empty_accumulator(self):
        weight_accumulator = dict()
        for name, data in self.model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)
        return weight_accumulator

    def sample_users_for_round(self, epoch) -> List[FLUser]:
        # if self.params.current_acc_poison < 0.8:
        # if True:
        #     # print(f"Round {epoch} when acc_clean > 0.6, all attackers involve in the FL training")
        #     # print(f"Round {epoch} when acc_poison < 0.8, all attackers involve in the FL training")
        #     print(f"Round {epoch} all attackers involve in the FL training")
        #     # Make sure users 0, 1, 2, 3 always participate in the training. The rest will be different from 0, 1, 2, 3
        #     sampled_ids = [0, 1, 2, 3]
        #     sampled_ids.extend(random.sample(
        #         range(4, self.params.fl_total_participants),
        #         self.params.fl_no_models - 4))
        # else:
        # sampled_ids = random.sample(
        #     range(self.params.fl_total_participants),
        #     self.params.fl_no_models)
        # if epoch < self.params.poison_epoch_stop + 1:
        #     sampled_ids = random.sample(
        #         range(self.params.fl_number_of_adversaries, self.params.fl_total_participants),
        #         self.params.fl_no_models - len(self.adversaries))
        
        #     sampled_ids.extend(self.adversaries)
        # else:
        #     sampled_ids = random.sample(
        #         range(self.params.fl_total_participants),
        #         self.params.fl_no_models)
        sampled_ids = random.sample(
                            range(self.params.fl_number_of_adversaries, self.params.fl_total_participants),
                            self.params.fl_no_models - len(self.adversaries))
        
        sampled_ids.extend(self.adversaries)
            
        random.shuffle(sampled_ids)

        sampled_users = []
        for pos, user_id in enumerate(sampled_ids):
            train_loader = self.fl_train_loaders[user_id]
            number_of_samples = self.fl_number_of_samples[user_id]
            compromised = self.check_user_compromised(epoch, pos, user_id)
            if compromised:
                logger.warning(f'Compromised user: {user_id}')
                user = FLUser(user_id, compromised=compromised,
                        train_loader=train_loader, backdoor_label=self.params.backdoor_label,
                        number_of_samples=number_of_samples)
            else:
                user = FLUser(user_id, compromised=compromised,
                        train_loader=train_loader, number_of_samples=number_of_samples)
            sampled_users.append(user)

        self.params.fl_round_participants = [user.user_id for user in sampled_users]
        total_samples = sum([user.number_of_samples for user in sampled_users])
        # self.params.fl_weight_contribution = [user.number_of_samples / total_samples for user in sampled_users]
        self.params.fl_weight_contribution = {user.user_id: user.number_of_samples / total_samples for user in sampled_users}
        self.params.fl_number_of_samples_each_user = {user.user_id: user.number_of_samples for user in sampled_users} # for implementing krum defense
        self.params.fl_local_updated_models = {}
        logger.warning(f"Sampled users for round {epoch}: {self.params.fl_weight_contribution}")
        logger.warning(f"Sampled users for round {epoch}: {self.params.fl_number_of_samples_each_user}")

        return sampled_users

    def check_user_compromised(self, epoch, pos, user_id):
        """Check if the sampled user is compromised for the attack.

        If single_epoch_attack is defined (eg not None) then ignore
        :param epoch:
        :param pos:
        :param user_id:
        :return:
        """
        compromised = False
        # if self.params.normal_training:
        #     print("Do normal training, without backdoor attack")
        #     return compromised
        # else:
        # print("Do backdoor training")
        if self.params.fl_single_epoch_attack is not None:
            if epoch == self.params.fl_single_epoch_attack:
                # if pos < self.params.fl_number_of_adversaries:
                if user_id == 0:
                    compromised = True
                    logger.warning(f'Attacking once at epoch {epoch}. Compromised'
                                f' user: {user_id}.')
        else:
            if epoch >= self.params.poison_epoch and epoch < self.params.poison_epoch_stop + 1:
                if self.params.fixed_frequency == 1: # Every epoch attack
                    compromised = user_id in self.adversaries
                else:
                    if epoch % self.params.fixed_frequency == 0: # Fixed frequency attack
                        compromised = user_id in self.adversaries
        return compromised

    def sample_adversaries(self) -> List[int]:
        adversaries_ids = []
        if self.params.fl_number_of_adversaries == 0:
            logger.warning(f'Running vanilla FL, no attack.')
        elif self.params.fl_single_epoch_attack is None:
            adversaries_ids = list(range(self.params.fl_number_of_adversaries))
            logger.warning(f'Attacking over multiple epochs with following '
                           f'users compromised: {adversaries_ids}.')
        else:
            logger.warning(f'Attack only on epoch: '
                           f'{self.params.fl_single_epoch_attack} with '
                           f'{self.params.fl_number_of_adversaries} compromised'
                           f' users.')

        return adversaries_ids

    def get_model_optimizer(self, model):
        local_model = deepcopy(model)
        local_model = local_model.to(self.params.device)

        optimizer = self.make_optimizer(local_model)

        return local_model, optimizer

    def copy_params(self, global_model: Module, local_model: Module):
        local_state = local_model.state_dict()
        for name, param in global_model.state_dict().items():
            if name in local_state and name not in self.ignored_weights:
                local_state[name].copy_(param)

    def update_global_model(self, weight_accumulator, global_model: Module):
        # self.last_global_model = deepcopy(self.model)
        for name, sum_update in weight_accumulator.items():
            if self.check_ignored_weights(name):
                continue
            # scale = self.params.fl_eta / self.params.fl_total_participants
            # average_update = scale * sum_update
            average_update = sum_update
            model_weight = global_model.state_dict()[name]
            model_weight.add_(average_update)

    def check_ignored_weights(self, name) -> bool:
        for ignored in self.ignored_weights:
            if ignored in name:
                return True

        return False
    
    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: dataset_classes, a preprocessed class-indices dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as
            parameters for
            dirichlet distribution to sample number of images in each class.
        """

        dataset_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if label in dataset_classes:
                dataset_classes[label].append(ind)
            else:
                dataset_classes[label] = [ind]
        class_size = len(dataset_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(dataset_classes.keys())

        for n in range(no_classes):
            random.shuffle(dataset_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = dataset_classes[n][
                               :min(len(dataset_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                dataset_classes[n] = dataset_classes[n][
                                   min(len(dataset_classes[n]), no_imgs):]

        return per_participant_list

    def sample_dirichlet_train_data_new(self, no_participants, alpha=0.9): # for chestxray
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: dataset_classes, a preprocessed class-indices dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as
            parameters for
            dirichlet distribution to sample number of images in each class.
        """

        dataset_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if label in dataset_classes:
                dataset_classes[label].append(ind)
            else:
                dataset_classes[label] = [ind]
        per_participant_list = defaultdict(list)
        no_classes = len(dataset_classes.keys())

        # Ensure each participant gets at least one image
        for user in range(no_participants):
            # Choose a random class
            class_choice = random.choice(list(dataset_classes.keys()))
            # Remove an image from the chosen class and assign it to the participant
            per_participant_list[user].append(dataset_classes[class_choice].pop())
            # If a class is empty, remove it from the dictionary
            if len(dataset_classes[class_choice]) == 0:
                del dataset_classes[class_choice]

        # Distribute the remaining images
        for n in list(dataset_classes.keys()):
            random.shuffle(dataset_classes[n])
            sampled_probabilities = len(dataset_classes[n]) * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = dataset_classes[n][:min(len(dataset_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                dataset_classes[n] = dataset_classes[n][min(len(dataset_classes[n]), no_imgs):]

        return per_participant_list

    def get_train(self, indices):
        """
        This method is used along with Dirichlet distribution
        :param indices:
        :return:
        """
        # train_loader = DataLoader(self.train_dataset,
        #                           batch_size=self.params.batch_size,
        #                           sampler=SubsetRandomSampler(
        #                               indices), drop_last=True) # Bug happens when a client has the number of training examples < train batch size
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.params.batch_size,
                                  sampler=SubsetRandomSampler(
                                      indices), drop_last=False)
        return train_loader, len(indices)

    def get_train_old(self, all_range, model_no):
        """
        This method equally splits the dataset.
        :param all_range:
        :param model_no:
        :return:
        """

        data_len = int(
            len(self.train_dataset) / self.params.fl_total_participants)
        sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
        # print("DEBUG: ", len(sub_indices))
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.params.batch_size,
                                  sampler=SubsetRandomSampler(
                                      sub_indices))
        return train_loader, len(sub_indices)
    
    def adding_local_updated_model(self, local_update: Dict[str, torch.Tensor], epoch=None, user_id=None):
        self.params.fl_local_updated_models[user_id] = local_update

    def get_iid_train_loader(self, all_range, model_no):
        """
        This method equally splits the dataset.
        :param all_range:
        :param model_no:
        :return:
        """

        data_len = int(
            len(self.train_dataset) / self.params.fl_total_participants)
        sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
        # print("DEBUG: ", len(sub_indices))
        # train_loader = DataLoader(self.train_dataset,
        #                           batch_size=self.params.batch_size,
        #                           sampler=SubsetRandomSampler(
        #                               sub_indices))
        train_loader = DataLoader(Subset(self.train_dataset, sub_indices),  # Use Subset to create a subset of the dataset
            batch_size=self.params.batch_size,
            shuffle=True)
        

        return train_loader
    
    def get_atkmodel(self):
        if self.params.task.lower() == 'cifar10':
            from attack_models.autoencoders import ConditionalAutoencoder, Autoencoder, MNISTConditionalAutoencoder
            from attack_models.unet import UNet, ConditionalUNet

            input_dim = self.params.input_shape[1]
            n_classes = self.params.num_classes
            pattern_tensor = torch.tensor([
                            [1., 0., 1.],
                            [-10., 1., -10.],
                            [-10., -10., 0.],
                            [-10., 1., -10.],
                            [1., 0., 1.]
                        ])
            
            # atkmodel = ConditionalAutoencoder(n_classes, input_dim, pattern_tensor).to(self.params.device)
            # atkmodel = ConditionalAutoencoder(n_classes, input_dim).to(self.params.device)
            atkmodel = ConditionalUNet(n_classes, input_dim, 3).to(self.params.device)
            # atkmodel = MNISTConditionalAutoencoder(n_classes, input_dim).to(self.params.device)
            # atkmodel = UNet(n_classes, input_dim, 3).to(self.params.device)
            # atkmodel = Autoencoder().to(self.params.device)
            # atkmodel = UNet(3).to(self.params.device)

            # tgtmodel = ConditionalAutoencoder(n_classes, input_dim, pattern_tensor).to(self.params.device)
            # tgtmodel = ConditionalAutoencoder(n_classes, input_dim).to(self.params.device)
            tgtmodel = ConditionalUNet(n_classes, input_dim, 3).to(self.params.device)
            # tgtmodel = MNISTConditionalAutoencoder(n_classes, input_dim).to(self.params.device)
            # tgtmodel = UNet(n_classes, input_dim, 3).to(self.params.device)
            # tgtmodel = Autoencoder().to(self.params.device)
            # tgtmodel = UNet(3).to(self.params.device)
            tgtmodel.load_state_dict(atkmodel.state_dict(), strict=True)

            # tgtoptimizer = torch.optim.Adam(tgtmodel.parameters(), lr=self.params.lr_atk)
            tgtoptimizer = torch.optim.Adam(tgtmodel.parameters(), lr=self.params.lr_atk, betas=(0.5, 0.999)) # Starts from exp67

            return atkmodel, tgtmodel, tgtoptimizer
        elif self.params.task.lower() == 'cifar100':
            from attack_models.autoencoders import ConditionalAutoencoder, Autoencoder, MNISTConditionalAutoencoder
            from attack_models.unet import UNet, ConditionalUNet

            input_dim = self.params.input_shape[1]
            n_classes = self.params.num_classes
            
            # atkmodel = ConditionalAutoencoder(n_classes, input_dim, pattern_tensor).to(self.params.device)
            # atkmodel = ConditionalAutoencoder(n_classes, input_dim).to(self.params.device)
            atkmodel = ConditionalUNet(n_classes, input_dim, 3).to(self.params.device)
            # atkmodel = MNISTConditionalAutoencoder(n_classes, input_dim).to(self.params.device)
            # atkmodel = UNet(n_classes, input_dim, 3).to(self.params.device)
            # atkmodel = Autoencoder().to(self.params.device)
            # atkmodel = UNet(3).to(self.params.device)

            # tgtmodel = ConditionalAutoencoder(n_classes, input_dim, pattern_tensor).to(self.params.device)
            # tgtmodel = ConditionalAutoencoder(n_classes, input_dim).to(self.params.device)
            tgtmodel = ConditionalUNet(n_classes, input_dim, 3).to(self.params.device)
            # tgtmodel = MNISTConditionalAutoencoder(n_classes, input_dim).to(self.params.device)
            # tgtmodel = UNet(n_classes, input_dim, 3).to(self.params.device)
            # tgtmodel = Autoencoder().to(self.params.device)
            # tgtmodel = UNet(3).to(self.params.device)
            tgtmodel.load_state_dict(atkmodel.state_dict(), strict=True)

            # tgtoptimizer = torch.optim.Adam(tgtmodel.parameters(), lr=self.params.lr_atk)
            tgtoptimizer = torch.optim.Adam(tgtmodel.parameters(), lr=self.params.lr_atk, betas=(0.5, 0.999)) # Starts from exp67

            return atkmodel, tgtmodel, tgtoptimizer
        elif self.params.task.lower() == 'mnist':
            from attack_models.autoencoders import MNISTConditionalAutoencoder
            from attack_models.unet import UNet, MNISTConditionalUNet

            input_dim = self.params.input_shape[1]
            n_classes = self.params.num_classes
            
            # atkmodel = ConditionalAutoencoder(n_classes, input_dim, pattern_tensor).to(self.params.device)
            # atkmodel = ConditionalAutoencoder(n_classes, input_dim).to(self.params.device)
            atkmodel = MNISTConditionalAutoencoder(n_classes, input_dim).to(self.params.device)
            # atkmodel = MNISTConditionalUNet(n_classes, input_dim, 1).to(self.params.device)
            # atkmodel = UNet(n_classes, input_dim, 3).to(self.params.device)
            # atkmodel = Autoencoder().to(self.params.device)
            # atkmodel = UNet(3).to(self.params.device)

            # tgtmodel = ConditionalAutoencoder(n_classes, input_dim, pattern_tensor).to(self.params.device)
            # tgtmodel = ConditionalAutoencoder(n_classes, input_dim).to(self.params.device)
            tgtmodel = MNISTConditionalAutoencoder(n_classes, input_dim).to(self.params.device)
            # tgtmodel = MNISTConditionalUNet(n_classes, input_dim, 1).to(self.params.device)
            # tgtmodel = UNet(n_classes, input_dim, 3).to(self.params.device)
            # tgtmodel = Autoencoder().to(self.params.device)
            # tgtmodel = UNet(3).to(self.params.device)
            tgtmodel.load_state_dict(atkmodel.state_dict(), strict=True)

            # tgtoptimizer = torch.optim.Adam(tgtmodel.parameters(), lr=self.params.lr_atk)
            tgtoptimizer = torch.optim.Adam(tgtmodel.parameters(), lr=self.params.lr_atk, betas=(0.5, 0.999)) # Starts from exp67

            return atkmodel, tgtmodel, tgtoptimizer
        elif self.params.task.lower() == 'chestxray':
            from attack_models.autoencoders import ChestXRayConditionalAutoencoder, ConditionalAutoencoder
            from attack_models.unet import UNet, ConditionalUNet, ChestXRayConditionalUNet

            input_dim = self.params.input_shape[1]
            n_classes = self.params.num_classes
            
            # atkmodel = ConditionalAutoencoder(n_classes, input_dim, pattern_tensor).to(self.params.device)
            # atkmodel = ConditionalAutoencoder(n_classes, input_dim).to(self.params.device)
            # atkmodel = ChestXRayConditionalAutoencoder(n_classes, input_dim).to(self.params.device)
            # atkmodel = ConditionalUNet(n_classes, input_dim, 3).to(self.params.device)
            # atkmodel = ConditionalUNet(n_classes, input_dim, 1).to(self.params.desvice) # chestxray
            atkmodel = ChestXRayConditionalUNet(n_classes, input_dim, 1).to(self.params.device) # chestxray
            # atkmodel = Autoencoder().to(self.params.device)
            # atkmodel = UNet(3).to(self.params.device)

            # tgtmodel = ConditionalAutoencoder(n_classes, input_dim, pattern_tensor).to(self.params.device)
            # tgtmodel = ConditionalAutoencoder(n_classes, input_dim).to(self.params.device)
            # tgtmodel = ChestXRayConditionalAutoencoder(n_classes, input_dim).to(self.params.device)
            # tgtmodel = ConditionalUNet(n_classes, input_dim, 3).to(self.params.device)
            # tgtmodel = ConditionalUNet(n_classes, input_dim, 1).to(self.params.device) # chestxray
            tgtmodel = ChestXRayConditionalUNet(n_classes, input_dim, 1).to(self.params.device) # chestxray
            # tgtmodel = Autoencoder().to(self.params.device)
            # tgtmodel = UNet(3).to(self.params.device)
            tgtmodel.load_state_dict(atkmodel.state_dict(), strict=True)

            # tgtoptimizer = torch.optim.Adam(tgtmodel.parameters(), lr=self.params.lr_atk)
            tgtoptimizer = torch.optim.Adam(tgtmodel.parameters(), lr=self.params.lr_atk, betas=(0.5, 0.999)) # Starts from exp67

            return atkmodel, tgtmodel, tgtoptimizer
        elif self.params.task.lower() == 'tinyimagenet':
            from attack_models.autoencoders import ChestXRayConditionalAutoencoder, ConditionalAutoencoder
            from attack_models.unet import UNet, ConditionalUNet
            from attack_models.unet_model import ConditionalUNet_v2

            input_dim = self.params.input_shape[1]
            n_classes = self.params.num_classes
            
            # atkmodel = ConditionalAutoencoder(n_classes, input_dim, pattern_tensor).to(self.params.device)
            # atkmodel = ConditionalAutoencoder(n_classes, input_dim).to(self.params.device) # tinyimagenet
            # atkmodel = ChestXRayConditionalAutoencoder(n_classes, input_dim).to(self.params.device)
            atkmodel = ConditionalUNet(n_classes, input_dim, 3).to(self.params.device) # tinyimagenet
            # atkmodel = ConditionalUNet_v2(n_classes, input_dim, 3).to(self.params.device) # tinyimagenet
            # checkpoint = torch.load("./pretrained/tinyimagenet_model_backdoor_epoch_100.pt.tar", map_location="cuda")
            # atkmodel.load_state_dict(checkpoint['tgt_state_dict'])

            # checkpoint = torch.load("./pretrained/tinyimagenet_model_backdoor_64_epoch_100.pt.tar", map_location="cuda")
            # atkmodel.load_state_dict(checkpoint['tgt_state_dict'])
            
            # atkmodel = ConditionalUNet(n_classes, input_dim, 1).to(self.params.device) # chestxray
            # atkmodel = Autoencoder().to(self.params.device)
            # atkmodel = UNet(3).to(self.params.device)

            # tgtmodel = ConditionalAutoencoder(n_classes, input_dim, pattern_tensor).to(self.params.device)
            # tgtmodel = ConditionalAutoencoder(n_classes, input_dim).to(self.params.device) # tinyimagenet
            # tgtmodel = ChestXRayConditionalAutoencoder(n_classes, input_dim).to(self.params.device)
            tgtmodel = ConditionalUNet(n_classes, input_dim, 3).to(self.params.device) # tinyimagenet
            # tgtmodel = ConditionalUNet_v2(n_classes, input_dim, 3).to(self.params.device) # tinyimagenet
            # tgtmodel = ConditionalUNet(n_classes, input_dim, 1).to(self.params.device) # chestxray
            # tgtmodel = Autoencoder().to(self.params.device)
            # tgtmodel = UNet(3).to(self.params.device)
            tgtmodel.load_state_dict(atkmodel.state_dict(), strict=True)

            # tgtoptimizer = torch.optim.Adam(tgtmodel.parameters(), lr=self.params.lr_atk)
            tgtoptimizer = torch.optim.Adam(tgtmodel.parameters(), lr=self.params.lr_atk, betas=(0.5, 0.999)) # Starts from exp67

            return atkmodel, tgtmodel, tgtoptimizer
        elif self.params.task.lower() == 'fashionmnist':
            from attack_models.autoencoders import FashionMNISTConditionalAutoencoder
            from attack_models.unet import UNet, FashionMNISTConditionalUNet

            input_dim = self.params.input_shape[1]
            n_classes = self.params.num_classes
            
            # atkmodel = ConditionalAutoencoder(n_classes, input_dim, pattern_tensor).to(self.params.device)
            # atkmodel = ConditionalAutoencoder(n_classes, input_dim).to(self.params.device)
            atkmodel = FashionMNISTConditionalAutoencoder(n_classes, input_dim).to(self.params.device)
            # atkmodel = FashionMNISTConditionalUNet(n_classes, input_dim, 1).to(self.params.device)
            # atkmodel = UNet(n_classes, input_dim, 3).to(self.params.device)
            # atkmodel = Autoencoder().to(self.params.device)
            # atkmodel = UNet(3).to(self.params.device)

            # tgtmodel = ConditionalAutoencoder(n_classes, input_dim, pattern_tensor).to(self.params.device)
            # tgtmodel = ConditionalAutoencoder(n_classes, input_dim).to(self.params.device)
            tgtmodel = FashionMNISTConditionalAutoencoder(n_classes, input_dim).to(self.params.device)
            # tgtmodel = FashionMNISTConditionalUNet(n_classes, input_dim, 1).to(self.params.device)
            # tgtmodel = UNet(n_classes, input_dim, 3).to(self.params.device)
            # tgtmodel = Autoencoder().to(self.params.device)
            # tgtmodel = UNet(3).to(self.params.device)
            tgtmodel.load_state_dict(atkmodel.state_dict(), strict=True)

            # tgtoptimizer = torch.optim.Adam(tgtmodel.parameters(), lr=self.params.lr_atk)
            tgtoptimizer = torch.optim.Adam(tgtmodel.parameters(), lr=self.params.lr_atk, betas=(0.5, 0.999)) # Starts from exp67

            return atkmodel, tgtmodel, tgtoptimizer
        else:
            raise NotImplementedError

    def target_transform(self):
        return lambda x: torch.ones_like(x) * self.params.backdoor_label
    
    def sample_negative_labels(self, label, n_classes):
        label_cpu = label.detach().cpu().numpy()
        neg_label = [np.random.choice([e for e in range(n_classes) if e != l], 1)[0] for l in label_cpu]
        neg_label = torch.tensor(np.array(neg_label))
        return neg_label.to(self.params.device)

