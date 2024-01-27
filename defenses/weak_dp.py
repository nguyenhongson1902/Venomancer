import torch

import logging
import os
import copy

from defenses.fedavg import FedAvg


logger = logging.getLogger('logger')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Weak_DP defense aka AddNoise
class Weak_DP(FedAvg):
    ignored_weights = ['num_batches_tracked'] #['tracked', 'running']

    def get_fl_weight(self, global_model):
        global_update = dict()
        for name, data in global_model.state_dict().items():
            if self.check_ignored_weights(name):
                continue
            global_update[name] = data

        return global_update
    
    def check_ignored_weights(self, name):
        for ignored in self.ignored_weights:
            if ignored in name:
                return True
        return False
    
    def load_model_weight(self, net, weight):
        index_bias = 0
        for name, data in net.state_dict().items():
            if self.check_ignored_weights(name):
                continue
            net.state_dict()[name].copy_(weight[index_bias:index_bias+data.numel()].view(data.size()))
            index_bias += data.numel()

    def add_noise_to_weights(self, global_model):
        global_weight = self.get_fl_weight(global_model)
        vectorized_weight = torch.cat([global_weight[name].view(-1) for name in global_weight.keys()])
        gaussian_noise = torch.randn(vectorized_weight.size(),
                            device=self.params.device) * self.params.stddev
        dp_weight = vectorized_weight + gaussian_noise
        self.load_model_weight(global_model, dp_weight)
        # logger.info("Weak DP Defense: added noise of norm: {}".format(torch.norm(gaussian_noise)))
        

        
            