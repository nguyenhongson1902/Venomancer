import torch
import numpy as np

import logging
import os
import copy

from defenses.fedavg import FedAvg


logger = logging.getLogger('logger')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Krum defense, select one client (mode krum) or multiple clients (mode multi-krum) to send the local updates to server
class RLR(FedAvg):
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
    
    def vectorize_net(self, net):
        weight = []
        for name, data in net.state_dict().items():
            if self.check_ignored_weights(name):
                continue
            weight.append(data.view(-1))
        return torch.cat(weight)
    
    def vectorize_net_params(self, net_params):
        # net_params: dict
        weight = []
        for name, data in net_params.items():
            if self.check_ignored_weights(name):
                continue
            weight.append(data.view(-1))
        return torch.cat(weight)

    def run(self, global_model, participated_clients):
        # participated_clients: key: user_id, value: client_model
        assert self.params.defense.lower() == "rlr", "rlr is not passed properly"
        
        # HYPERPARAMETERS of RLR (Change this manually)
        robustLR_threshold, aggr, noise, clip, server_lr = 2, 'avg', 0, 0, self.params.lr

        n_params = sum(p.numel() for p in global_model.parameters())
        lr_vector = torch.Tensor([server_lr]*n_params).to(self.params.device)
    
        local_updates = {user_id: self.vectorize_net_params(client_params).detach().cpu().numpy() for user_id, client_params in self.params.fl_local_updated_models.items()} # key: user_id, value: local_update (dict)
        aggr_freq = self.params.fl_weight_contribution # key: user_id, value: weight (float)
        
        if robustLR_threshold > 0:
            lr_vector = self.compute_robustLR(local_updates, server_lr, robustLR_threshold)
        
        aggregated_updates = 0
        if aggr=='avg':      
            aggregated_updates = self.agg_avg(local_updates, aggr_freq)
        elif aggr =='comed':
            aggregated_updates = self.agg_comed(local_updates) # TODO
        elif aggr == 'sign':
            aggregated_updates = self.agg_sign(local_updates) # TODO
            
        if noise > 0:
            aggregated_updates.add_(torch.normal(mean=0, std=noise*clip, size=(n_params,)).to(self.params.device))

        cur_global_params = self.vectorize_net(global_model).detach().cpu().numpy()
        new_global_params = (cur_global_params + lr_vector*aggregated_updates).astype(np.float32)
        
        user_id_aggregated = list(participated_clients.keys())[0]
        aggregated_model = participated_clients[user_id_aggregated]
        self.load_model_weight(aggregated_model, torch.from_numpy(new_global_params).to(self.params.device))

        self.params.fl_weight_contribution = {user_id_aggregated: 1.0}
        self.params.fl_local_updated_models = {user_id_aggregated: self.params.fl_local_updated_models[user_id_aggregated]}

    def compute_robustLR(self, agent_updates, server_lr, robustLR_threshold):
        agent_updates_sign = [np.sign(update) for user_id, update in agent_updates.items()]  
        sm_of_signs = np.abs(sum(agent_updates_sign))
        print(f"sm_of_signs is: {sm_of_signs}")
        
        sm_of_signs[sm_of_signs < robustLR_threshold] = -server_lr
        sm_of_signs[sm_of_signs >= robustLR_threshold] = server_lr                                            
        return sm_of_signs
    
    def agg_avg(self, agent_updates_dict, num_dps):
        """ classic fed avg """
        assert set(agent_updates_dict.keys()) == set(num_dps), "user_id doesn't match"
        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = num_dps[_id]
            sm_updates +=  n_agent_data * update
            total_data += n_agent_data
        return  sm_updates / total_data
    
    def agg_comed(self, agent_updates_dict):
        agent_updates_col_vector = [update.view(-1, 1) for update in agent_updates_dict.values()]
        concat_col_vectors = torch.cat(agent_updates_col_vector, dim=1)
        return torch.median(concat_col_vectors, dim=1).values
    
    def agg_sign(self, agent_updates_dict):
        """ aggregated majority sign update """
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_signs = torch.sign(sum(agent_updates_sign))
        return torch.sign(sm_signs)