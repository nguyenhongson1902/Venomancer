import torch
import numpy as np

import logging
import os
import copy

from defenses.fedavg import FedAvg


logger = logging.getLogger('logger')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Krum defense, select one client (mode krum) or multiple clients (mode multi-krum) to send the local updates to server
class Krum(FedAvg):
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

    def find_smallest_neighbors(self, participated_clients):
        # find the smallest neighbors for each client
        # participated_clients: dict of user_id: client_model
        weight_contribution = self.params.fl_weight_contribution
        vectorize_nets = {user_id: self.vectorize_net(client_model).detach().cpu().numpy() for user_id, client_model in participated_clients.items()}

        # nb_in_score = self.num_workers-self.s-2
        # scores = []
        # for i, g_i in enumerate(vectorize_nets):
        #     dists = []
        #     for j, g_j in enumerate(vectorize_nets):
        #         if j == i:
        #             continue
        #         if j < i:
        #             dists.append(neighbor_distances[j][i - j - 1])
        #         else:
        #             dists.append(neighbor_distances[i][j - i - 1])
        #     # alternative to topk in pytorch and tensorflow
        #     topk_ind = np.argpartition(dists, nb_in_score)[:nb_in_score]
        #     scores.append(sum(np.take(dists, topk_ind)))

        nb_in_score = self.params.fl_no_models - self.params.fl_number_of_adversaries - 2
        
        # TODO: Double check
        '''
        vectorized_nets = [weights1, weights2, weights3, weights4, weights5, weights6], index = [0, 1, 2, 3, 4, 5]
        neighbor_distances = [[1, 2, 3], [4, 5], [6], [5, 2, 4, 3], [6, 7], [8, 10, 11]], index = [0, 1, 2, 3, 4, 5]

        '''

        
            