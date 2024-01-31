import torch
import numpy as np

import logging
import os
import copy

from defenses.fedavg import FedAvg


logger = logging.getLogger('logger')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# RFA defense
# TODO: Refactor RFA defense from Attack of the tails
class RFA(FedAvg):
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

    def run(self, participated_clients):
        # find the smallest neighbors for each client
        # participated_clients: key: user_id, value: client_model (architecture)
        assert self.params.mode_krum in ["krum", "multi_krum"], "mode_krum is not either krum or multi_krum"
        vectorize_nets = {user_id: self.vectorize_net(client_model).detach().cpu().numpy() for user_id, client_model in participated_clients.items()}

        nb_in_score = self.params.fl_no_models - self.params.fl_number_of_adversaries - 2
        
        neighbor_distances = dict()
        for user_id, vectorized_net in vectorize_nets.items():
            neighbor_distances[user_id] = []
            for neighbor_id, neighbor_vectorized_net in vectorize_nets.items():
                if user_id == neighbor_id:
                    continue
                neighbor_distances[user_id].append(np.linalg.norm(vectorized_net - neighbor_vectorized_net)**2)
        
        scores = dict()
        for user_id, distances in neighbor_distances.items():
            topk_ind = np.argpartition(distances, nb_in_score)[:nb_in_score]
            scores[user_id] = sum(np.take(distances, topk_ind))

        if self.params.mode_krum == "krum":
            user_id_star = min(scores, key=scores.get)
            self.params.fl_weight_contribution = {user_id_star: 1.0}
            self.params.fl_local_updated_models = {user_id_star: self.params.fl_local_updated_models[user_id_star]}
        elif self.params.mode_krum == "multi_krum":
            keys, values = np.array(list(scores.keys())), np.array(list(scores.values()))
            topk_ind = np.argpartition(values, nb_in_score+2)[:nb_in_score+2]
            users_to_take_average = keys[topk_ind]

            total_samples = 0
            for user_id, n_samples in self.params.fl_number_of_samples_each_user.items():
                if user_id in users_to_take_average:
                    total_samples += n_samples

            new_weight_contribution = {}
            for user_id, n_samples in self.params.fl_number_of_samples_each_user.items():
                if user_id in users_to_take_average:
                    new_weight_contribution[user_id] = n_samples / total_samples
            
            weighted_params = []
            for user_id, vectorize_net in vectorize_nets.items():
                if user_id in users_to_take_average:
                    weighted_params.append(vectorize_net * new_weight_contribution[user_id])
            
            weighted_params = np.array(weighted_params)
            aggregated_params = weighted_params.sum(axis=0)

            user_id_aggregated = list(participated_clients.keys())[0]
            aggregated_model = participated_clients[user_id_aggregated]
            self.load_model_weight(aggregated_model, torch.from_numpy(aggregated_params).to(self.params.device))

            self.params.fl_weight_contribution = {user_id_aggregated: 1.0}
            self.params.fl_local_updated_models = {user_id_aggregated: self.params.fl_local_updated_models[user_id_aggregated]}

            

            



        '''
        vectorized_nets = [weights1, weights2, weights3, weights4, weights5, weights6], index = [0, 1, 2, 3, 4, 5]
        neighbor_distances = [[1, 2, 3], [4, 5], [6], [5, 2, 4, 3], [6, 7], [8, 10, 11]], index = [0, 1, 2, 3, 4, 5]

        '''

        
            