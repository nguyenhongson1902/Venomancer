import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from typing import List
from copy import deepcopy


class CustomDataset(Dataset): # for soft labels
    def __init__(self, original_dataset, indices, new_labels):
        self.original_dataset = original_dataset
        self.indices = indices
        self.new_labels = new_labels
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        data, _ = self.original_dataset[self.indices[idx]]
        label = self.new_labels[idx]
        return data, label


class KnowledgeDistiller:
    """
    A class for Knowledge Distillation using ensembles.
    """

    def __init__(
        self,
        dataset,
        epochs=2,
        batch_size=16,
        temperature=1,
        method="avglogits",
        device="cpu"
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.T = temperature
        self.epochs = epochs
        self.lr = 0.0001
        self.momentum = 0.5
        self.swa_lr = 0.005
        self.method = method
        self.device = device
        # self.Optim = optim.SGD
        # self.Loss = nn.KLDivLoss
        # self.malClients = malClients

    def distill_knowledge(self, teacher_ensemble, student_model):
        """
        Takes in a teacher ensemble (list of models) and a student model.
        Trains the student model using unlabelled dataset, then returns it.
        Args:
            teacher_ensemble is list of models used to construct pseudolabels using self.method
            student_model is models that will be trained
        """
        # Set labels as soft ensemble prediction
        original_dataset = self.dataset.dataset
        subset_indices = self.dataset.indices
        soft_labels = self.pseudolabels_from_ensemble(teacher_ensemble, student_model, self.method)
        custom_dataset = CustomDataset(original_dataset, subset_indices, soft_labels)

        opt = optim.SGD(
            student_model.parameters(), momentum=self.momentum, lr=self.lr, weight_decay=1e-4
        )
        Loss = nn.KLDivLoss
        loss = Loss(reduction="batchmean")

        # https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
        swa_model = AveragedModel(student_model).to(self.device)
        scheduler = CosineAnnealingLR(opt, T_max=100)
        swa_scheduler = SWALR(opt, swa_lr=self.swa_lr)

        dataloader = DataLoader(custom_dataset, batch_size=self.batch_size)
        for i in range(self.epochs):
            total_err = 0
            for j, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                pred = student_model(x)
                err = loss(F.log_softmax(pred / self.T, dim=1), y) * self.T * self.T
                err.backward()
                total_err += err
                opt.step()
            print(f"KD epoch {i}: {total_err}")
            scheduler.step()
            swa_model.update_parameters(student_model)
            swa_scheduler.step()

            torch.optim.swa_utils.update_bn(dataloader, swa_model, device=self.device)

        return swa_model.module

    def pseudolabels_from_ensemble(self, ensemble, student_model, method=None):
        """
        Combines the probabilities to make ensemble predictions.
        3 possibile methods:
            avglogits: Takes softmax of the average outputs of the models
            medlogits: Takes softmax of the median outputs of the models
            avgprob: Averages the softmax of the outputs of the models

        Idea: Use median instead of averages for the prediction probabilities!
            This might make the knowledge distillation more robust to confidently bad predictors.
        """
        if method is None:
            method = self.method

        local_model = deepcopy(student_model)

        with torch.no_grad():
            distill_dataloader = DataLoader(self.dataset, batch_size=self.batch_size)
            preds = []
            for i, (x, y) in enumerate(distill_dataloader):
                x = x.to(self.device)
                tmp = []
                for user_id, local_update in ensemble.items():

                    local_model.load_state_dict(local_update)
                    output = local_model(x) / self.T
                    tmp.append(output)
                preds_batch = torch.stack(tmp)
                preds.append(preds_batch)
            preds = torch.cat(preds, dim=1)
            print(f"Final preds shape: {preds.shape}")
            
            if method == "avglogits":
                pseudolabels = preds.mean(dim=0)
                return F.softmax(pseudolabels, dim=1)

            elif method == "medlogits":
                pseudolabels, idx = preds.median(dim=0)
                return F.softmax(pseudolabels, dim=1)

            elif method == "avgprob":
                preds = F.softmax(preds, dim=2)
                pseudolabels = preds.mean(dim=0)
                return pseudolabels

            else:
                raise ValueError(
                    "pseudolabel method should be one of: avglogits, medlogits, avgprob"
                )


    def median_based_scores(self, ensemble, global_model, weight_contribution):
        """
        Gives scores reative to how often the models had the median logit.
        """
        print(f"Calculating model scores based on frequency of median logits")
        local_model = deepcopy(global_model)

        with torch.no_grad():
            client_p = [] # original weights computed based on #data on each client
            for user_id, local_update in ensemble.items():
                client_p.append(weight_contribution[user_id])

            distill_dataloader = DataLoader(self.dataset, batch_size=self.batch_size)
            preds = []
            for i, (x, y) in enumerate(distill_dataloader):
                x = x.to(self.device)
                tmp = []
                for user_id, local_update in ensemble.items():

                    local_model.load_state_dict(local_update)
                    output = local_model(x) / self.T
                    tmp.append(output)
                preds_batch = torch.stack(tmp)
                preds.append(preds_batch)
            preds = torch.cat(preds, dim=1)
            pseudolabels, idx = preds.median(dim=0)

            counts = torch.bincount(idx.view(-1), minlength=preds.size(0)).to(self.device)
            counts_p = counts / counts.sum()

            client_p = torch.tensor(client_p).to(self.device)
            counts_p *= client_p
            counts_p /= counts_p.sum()

            # Update self.params.fl_weight_contribution
            for i, user_id in enumerate(weight_contribution.keys()):
                weight_contribution[user_id] = counts_p[i]

        # return counts_p


        # def loss_fn_kd(outputs, labels, teacher_outputs, temperature):
        # """
        # Compute the knowledge-distillation (KD) loss given outputs, labels.
        # "Hyperparameters": temperature and alpha
        # NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
        # and student expects the input tensor to be log probabilities! See Issue #2
        # """
        # alpha = 0.5
        # T = temperature
        # KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
        # F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
        # F.cross_entropy(outputs, labels) * (1. - alpha)
        # return KD_loss
