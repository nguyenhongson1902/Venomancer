import torch
import logging
import os
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch.utils.data
import torchvision.transforms as transforms
from torch.nn import Module
from defenses.fedavg import FedAvg
from utils.knowledge_distiller import KnowledgeDistiller

logger = logging.getLogger('logger')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class FedRAD(FedAvg):
    ignored_weights = ['num_batches_tracked']#['tracked', 'running']

    def aggr(self, weight_accumulator, global_model):
        distillation_dataset = self.params.server_dataset
        true_labels = [distillation_dataset.dataset.targets[i] for i in distillation_dataset.indices]

        kd = KnowledgeDistiller(distillation_dataset, method="medlogits", epochs=2, temperature=1, batch_size=16, device=self.params.device)

        
        kd.median_based_scores(ensemble=self.params.fl_local_updated_models, global_model=global_model, weight_contribution=self.params.fl_weight_contribution) # Update self.params.fl_weight_contribution

        for user_id, weight_contrib_user in self.params.fl_weight_contribution.items():
            loaded_params = self.params.fl_local_updated_models[user_id]
            self.accumulate_weights(weight_accumulator, \
                                    {key: (loaded_params[key] * weight_contrib_user).to(self.params.device) for \
                                        key in loaded_params})
            
        self.update_global_model(weight_accumulator, global_model) # After this, global model is averaged
        
        kd.distill_knowledge(teacher_ensemble=self.params.fl_local_updated_models, student_model=global_model) # After this, global model gets updated


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