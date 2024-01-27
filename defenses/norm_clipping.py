import torch

import logging
import os
import copy

from defenses.fedavg import FedAvg


logger = logging.getLogger('logger')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Norm_Clipping defense aka WeightDiffClippingDefense
class Norm_Clipping(FedAvg):

    def clip_weight_diff(self):
        fl_local_updated_models = copy.deepcopy(self.params.fl_local_updated_models)
        for user_id, local_update in fl_local_updated_models.items():
            flatten_weights = torch.cat([local_update[name].view(-1) for name in local_update.keys()])                
            weight_diff_norm = torch.norm(flatten_weights).item()

            for name in local_update.keys():
                local_update[name] = local_update[name]/max(1, weight_diff_norm/self.params.norm_bound)
            
            self.params.fl_local_updated_models[user_id] = local_update

        
            