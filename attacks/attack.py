import logging
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from synthesizers.synthesizer import Synthesizer
from attacks.loss_functions import compute_all_losses_and_grads
from utils.parameters import Params
import math
import time
import copy
import numpy as np
logger = logging.getLogger('logger')


class Attack:
    params: Params
    synthesizer: Synthesizer
    local_dataset: DataLoader
    loss_tasks: List[str]
    fixed_scales: Dict[str, float]
    ignored_weights = ['num_batches_tracked']#['tracked', 'running']

    def __init__(self, params, synthesizer):
        self.params = params
        self.synthesizer = synthesizer
        self.loss_tasks = ['normal', 'backdoor']
        self.fixed_scales = {'normal':0.5, 'backdoor':0.5}

        # FOR A3FL
        if self.params.attack_type == "a3fl":
            self.previous_global_model = None
            self.setup()


    def perform_attack(self, _) -> None:
        raise NotImplemented

    def compute_blind_loss(self, model, criterion, batch, attack, fixed_model=None):
        """

        :param model:
        :param criterion:
        :param batch:
        :param attack: Do not attack at all. Ignore all the parameters
        :return:
        """
        batch = batch.clip(self.params.clip_batch)
        loss_tasks = self.loss_tasks.copy() if attack else ['normal']
        # Should add the backdoor injection code here
        batch_back = self.synthesizer.make_backdoor_batch(batch, attack=attack)
        scale = dict()

        if len(loss_tasks) == 1:
            loss_values = compute_all_losses_and_grads(
                loss_tasks,
                self, model, criterion, batch, batch_back
            )
        else:
            loss_values = compute_all_losses_and_grads(
                loss_tasks,
                self, model, criterion, batch, batch_back,
                fixed_model = fixed_model)

            for t in loss_tasks:
                scale[t] = self.fixed_scales[t]

        if len(loss_tasks) == 1:
            scale = {loss_tasks[0]: 1.0}
        blind_loss = self.scale_losses(loss_tasks, loss_values, scale)

        return blind_loss

    def scale_losses(self, loss_tasks, loss_values, scale):
        blind_loss = 0
        for it, t in enumerate(loss_tasks):
            self.params.running_losses[t].append(loss_values[t].item())
            self.params.running_scales[t].append(scale[t])
            if it == 0:
                blind_loss = scale[t] * loss_values[t]
            else:
                blind_loss += scale[t] * loss_values[t]
        self.params.running_losses['total'].append(blind_loss.item())
        return blind_loss

    def scale_update(self, local_update: Dict[str, torch.Tensor], gamma):
        for name, value in local_update.items():
            value.mul_(gamma)

    def get_fl_update(self, local_model, global_model) -> Dict[str, torch.Tensor]:
        local_update = dict()
        for name, data in local_model.state_dict().items():
            if self.check_ignored_weights(name):
                continue
            local_update[name] = (data - global_model.state_dict()[name])

        return local_update
    
    def check_ignored_weights(self, name) -> bool:
        for ignored in self.ignored_weights:
            if ignored in name:
                return True

        return False

    def get_update_norm(self, local_update):
        squared_sum = 0
        for name, value in local_update.items():
            if 'tracked' in name or 'running' in name:
                continue
            squared_sum += torch.sum(torch.pow(value, 2)).item()
        update_norm = math.sqrt(squared_sum)
        return update_norm
    
    def compute_loss(self, model, criterion, batch, attack, fixed_model=None):
        batch = batch.clip(self.params.clip_batch)
        inputs, labels = batch.inputs, batch.labels
        # print("DEBUG: ", inputs.shape, labels.shape)
        outputs = model(inputs)

        # print("DEBUG: State Dict: ", model.state_dict())

        # print("DEBUG: ", outputs.shape)
        # print("DEBUG: ", outputs)
        loss = criterion(outputs, labels)
        loss = loss.mean()
        
        return loss
    
    # FOR A3FL
    def setup(self):
        self.handcraft_rnds = 0
        self.trigger = torch.ones((1,3,32,32), requires_grad=False, device = 'cuda')*0.5
        self.mask = torch.zeros_like(self.trigger)
        self.mask[:, :, 2:2+self.params.trigger_size, 2:2+self.params.trigger_size] = 1 # trigger_size = 5, 
        self.mask = self.mask.cuda()
        self.trigger0 = self.trigger.clone()

    # FOR A3FL
    def get_adv_model(self, model, dl, trigger, mask):
        adv_model = copy.deepcopy(model)
        adv_model.train()
        ce_loss = torch.nn.CrossEntropyLoss()
        adv_opt = torch.optim.SGD(adv_model.parameters(), lr = 0.01, momentum=0.9, weight_decay=5e-4)
        # for _ in range(self.helper.config.dm_adv_epochs): # dm_adv_epochs = 5
        for _ in range(self.params.dm_adv_epochs): # dm_adv_epochs = 5
            for inputs, labels in dl:
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = trigger*mask +(1-mask)*inputs
                outputs = adv_model(inputs)
                loss = ce_loss(outputs, labels)
                adv_opt.zero_grad()
                loss.backward()
                adv_opt.step()

        sim_sum = 0.
        sim_count = 0.
        cos_loss = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
        for name in dict(adv_model.named_parameters()):
            if 'conv' in name:
                sim_count += 1
                sim_sum += cos_loss(dict(adv_model.named_parameters())[name].grad.reshape(-1),\
                                    dict(model.named_parameters())[name].grad.reshape(-1))
        return adv_model, sim_sum/sim_count

    # FOR A3FL
    def search_trigger(self, model, dl, type_, adversary_id = 0, epoch = 1):
        trigger_optim_time_start = time.time()
        K = 0
        model.eval()
        adv_models = []
        adv_ws = []

        def val_asr(model, dl, t, m):
            ce_loss = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
            correct = 0.
            num_data = 0.
            total_loss = 0.
            with torch.no_grad():
                for inputs, labels in dl:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    inputs = t*m +(1-m)*inputs
                    # labels[:] = self.helper.config.target_class # target_class = 2
                    # labels[:] = self.params.backdoor_label # 1 target label
                    labels = self.sample_negative_labels(labels, n_classes=10) # multi-target
                    output = model(inputs)
                    loss = ce_loss(output, labels)
                    total_loss += loss
                    pred = output.data.max(1)[1] 
                    correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
                    num_data += output.size(0)
            asr = correct/num_data
            return asr, total_loss
        
        ce_loss = torch.nn.CrossEntropyLoss()
        # alpha = self.helper.config.trigger_lr # 0.01
        alpha = self.params.trigger_lr
        
        K = self.params.trigger_outter_epochs # 200, training iterations for trigger t
        t = self.trigger.clone()
        m = self.mask.clone()
        def grad_norm(gradients):
            grad_norm = 0
            for grad in gradients:
                grad_norm += grad.detach().pow(2).sum()
            return grad_norm.sqrt()
        ga_loss_total = 0.
        normal_grad = 0.
        ga_grad = 0.
        count = 0
        trigger_optim = torch.optim.Adam([t], lr = alpha*10, weight_decay=0)
        for iter in range(K):
            if iter % 10 == 0:
                asr, loss = val_asr(model, dl, t, m)
            if iter % self.params.dm_adv_K == 0 and iter != 0: # dm_adv_K = 1
                if len(adv_models)>0:
                    for adv_model in adv_models:
                        del adv_model
                adv_models = []
                adv_ws = [] # contains average cosine similarity between the gradients of the adversarial model and the original model
                # for _ in range(self.helper.config.dm_adv_model_count): # dm_adv_model_count = 1
                for _ in range(self.params.dm_adv_model_count): # dm_adv_model_count = 1
                    adv_model, adv_w = self.get_adv_model(model, dl, t, m) 
                    adv_models.append(adv_model)
                    adv_ws.append(adv_w)
            

            for inputs, labels in dl:
                count += 1
                t.requires_grad_()
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = t*m +(1-m)*inputs
                # labels[:] = self.helper.config.target_class
                # labels[:] = self.params.backdoor_label # 1 target label
                labels = self.sample_negative_labels(labels, n_classes=10) # multi-target
                outputs = model(inputs) 
                loss = ce_loss(outputs, labels)
                
                if len(adv_models) > 0:
                    for am_idx in range(len(adv_models)):
                        adv_model = adv_models[am_idx]
                        adv_w = adv_ws[am_idx]
                        outputs = adv_model(inputs)
                        nm_loss = ce_loss(outputs, labels)
                        if loss == None:
                            loss = self.params.noise_loss_lambda*adv_w*nm_loss/self.params.dm_adv_model_count
                        else:
                            loss += self.params.noise_loss_lambda*adv_w*nm_loss/self.params.dm_adv_model_count # noise_loss_lambda = 0.01
                if loss != None:
                    loss.backward()
                    normal_grad += t.grad.sum()
                    new_t = t - alpha*t.grad.sign()
                    t = new_t.detach_()
                    t = torch.clamp(t, min = -2, max = 2)
                    t.requires_grad_()
        t = t.detach()
        self.trigger = t
        self.mask = m
        trigger_optim_time_end = time.time()
    
    # FOR A3FL
    def poison_input(self, inputs, labels, eval=False):
        if eval:
            bkd_num = inputs.shape[0]
        else:
            bkd_num = int(self.params.bkd_ratio * inputs.shape[0])
        inputs[:bkd_num] = self.trigger*self.mask + inputs[:bkd_num]*(1-self.mask)
        # labels[:bkd_num] = self.helper.config.target_class
        # labels[:bkd_num] = self.params.backdoor_label # 1 target label
        labels[:bkd_num] = self.sample_negative_labels(labels[:bkd_num], n_classes=10) # multi-target
        return inputs, labels
    
    # FOR A3FL
    def sample_negative_labels(self, label, n_classes):
        label_cpu = label.detach().cpu().numpy()
        neg_label = [np.random.choice([e for e in range(n_classes) if e != l], 1)[0] for l in label_cpu]
        neg_label = torch.tensor(np.array(neg_label))
        return neg_label.to(self.params.device)