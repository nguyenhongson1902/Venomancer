import torch
import logging
import os
import random
import numpy as np
import sklearn.metrics.pairwise as smp
import hdbscan
from tqdm import tqdm
from copy import deepcopy
import torch.utils.data
import torchvision.transforms as transforms
from torch.nn import Module
from defenses.fedavg import FedAvg

logger = logging.getLogger('logger')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Deepsight(FedAvg):
    num_seeds: int = 3
    num_samples: int = 20000
    tau: float = 1/3

    def aggr(self, weight_accumulator, global_model: Module):
        num_channel = 3
        if 'mnist' == self.params.task.lower():
            # This is the default setting for MNIST
            dim = 28
            num_channel = 1
            
            # When applying Grayscale transform
            # dim = 32
        elif 'cifar10' == self.params.task.lower():
            dim = 32
        else:
            dim = 224
        # layer_name = 'fc2' if 'MNIST' in self.params.task else 'fc' # For SimpleNet
        # layer_name = 'linear9' if 'MNIST' in self.params.task else 'fc' # For NetC_MNIST
        layer_name = "linear" if "cifar10" == self.params.task.lower() else "fc"
        num_classes = 200 if 'imagenet' == self.params.task.lower() else 10

        # Threshold exceedings and NEUPs
        TEs, NEUPs, ed = [], [], []
        # for i in range(self.params.fl_no_models):
            # file_name = f'{self.params.folder_path}/saved_updates/update_{i}.pth'
            # loaded_params = torch.load(file_name)
        for user_id, local_update in self.params.fl_local_updated_models.items():
            loaded_params = local_update

            ed = np.append(ed, self.get_update_norm(loaded_params))
            UPs = abs(loaded_params[f'{layer_name}.bias'].cpu().numpy()) +\
                np.sum(abs(loaded_params[f'{layer_name}.weight'].cpu().numpy()), \
                axis=1)
            NEUP = UPs**2/np.sum(UPs**2)
            TE = 0
            for j in NEUP:
                if j >= (1/num_classes)*np.max(NEUP):
                    TE += 1
            NEUPs = np.append(NEUPs, NEUP)
            TEs.append(TE)
        logger.warning(f'Deepsight: Threshold Exceedings {TEs}')
        labels = []
        for i in TEs:
            if i >= np.median(TEs)/2:
                labels.append(False)
            else:
                labels.append(True)

        # ddif
        DDifs = []
        for i, seed in tqdm(enumerate(range(self.num_seeds))):
            torch.manual_seed(seed)
            dataset = NoiseDataset([num_channel, dim, dim], self.num_samples)
            loader = torch.utils.data.DataLoader(dataset, self.params.batch_size, shuffle=False)

            # for j in tqdm(range(self.params.fl_no_models)):
            #     file_name = f'{self.params.folder_path}/saved_updates/update_{j}.pth'
            #     loaded_params = torch.load(file_name)
            for user_id, local_update in tqdm(self.params.fl_local_updated_models.items()):
                loaded_params = local_update

                local_model = deepcopy(global_model)
                for name, data in loaded_params.items():
                    if self.check_ignored_weights(name):
                        continue
                    local_model.state_dict()[name].add_(data)

                local_model.eval()
                global_model.eval()
                DDif = torch.zeros(num_classes).to(self.params.device)
                for x in loader:
                    x = x.to(self.params.device)
                    with torch.no_grad():
                        output_local = local_model(x)
                        output_global = global_model(x)
                        if 'mnist' != self.params.task.lower():
                            output_local = torch.softmax(output_local, dim=1)
                            output_global = torch.softmax(output_global, dim=1)
                    temp = torch.div(output_local, output_global+1e-30) # avoid zero-value
                    temp = torch.sum(temp, dim=0)
                    DDif.add_(temp)

                DDif /= self.num_samples
                DDifs = np.append(DDifs, DDif.cpu().numpy())
        DDifs = np.reshape(DDifs, (self.num_seeds, self.params.fl_no_models, -1))
        logger.warning("Deepsight: Finish measuring DDif")

        # cosine distance
        # for i in range(self.params.fl_no_models):
        #     updates_name = f'{self.params.folder_path}/saved_updates/update_{i}.pth'
        #     loaded_params = torch.load(updates_name)
        local_params = []
        for user_id, local_update in self.params.fl_local_updated_models.items():
            loaded_params = local_update
            for name, data in loaded_params.items():
                if layer_name in name:
                    temp = local_model.state_dict()[name].cpu().numpy()
                    local_params = np.append(local_params, temp)
        cd = smp.cosine_distances(local_params.reshape(self.params.fl_no_models, -1))
        logger.warning("Deepsight: Finish calculating cosine distance")

        # classification
        cosine_clusters = hdbscan.HDBSCAN(metric='precomputed').fit_predict(cd)
        cosine_cluster_dists = dists_from_clust(cosine_clusters, self.params.fl_no_models)

        neup_clusters = hdbscan.HDBSCAN().fit_predict(np.reshape(NEUPs, (-1,1)))
        neup_cluster_dists = dists_from_clust(neup_clusters, self.params.fl_no_models)

        ddif_clusters, ddif_cluster_dists = [],[]
        for i in range(self.num_seeds):
            ddif_cluster_i = hdbscan.HDBSCAN().fit_predict(np.reshape(DDifs[i], (-1,1)))
            # ddif_clusters = np.append(ddif_clusters, ddif_cluster_i)
            ddif_cluster_dists = np.append(ddif_cluster_dists,
                dists_from_clust(ddif_cluster_i, self.params.fl_no_models))
        merged_ddif_cluster_dists = np.mean(np.reshape(ddif_cluster_dists,
                            (self.num_seeds, self.params.fl_no_models, self.params.fl_no_models)),
                            axis=0)
        merged_distances = np.mean([merged_ddif_cluster_dists,
                                    neup_cluster_dists,
                                    cosine_cluster_dists], axis=0)
        clusters = hdbscan.HDBSCAN(metric='precomputed').fit_predict(merged_distances)
        positive_counts = {}
        total_counts = {}
        ## TODO: DEBUG
        for i, c in enumerate(clusters):
            if c==-1:
                continue
            if c in positive_counts:
                positive_counts[c] += 1 if labels[i] else 0
                total_counts[c] += 1
            else:
                positive_counts[c] = 1 if labels[i] else 0
                total_counts[c] = 1
        logger.warning("Deepsight: Finish classification")

        # Aggregate and norm-clipping
        st = np.median(ed)
        print(f"Deepsight: clipping bound {st}")
        adv_clip = []
        discard_name = []
        for i, c in enumerate(clusters):
            if i < self.params.fl_number_of_adversaries:
                adv_clip.append(st/ed[i])
            if c!=-1 and positive_counts[c] / total_counts[c] < self.tau:
                file_name = f'{self.params.folder_path}/saved_updates/update_{i}.pth'
                loaded_params = torch.load(file_name)
                if 1 > st/ed[i]:
                    for name, data in loaded_params.items():
                        if self.check_ignored_weights(name):
                            continue
                        data.mul_(st/ed[i])
                self.accumulate_weights(weight_accumulator, loaded_params)
            else:
                discard_name.append(i)
        logger.warning(f"Deepsight: Discard update from client {discard_name}")
        logger.warning(f"Deepsight: clip for adv {adv_clip}")
        return weight_accumulator

class NoiseDataset(torch.utils.data.Dataset):
    def __init__(self, size, num_samples):
        self.size = size
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        noise = torch.rand(self.size)
        return noise

def dists_from_clust(clusters, N):
    pairwise_dists = np.ones((N,N))
    for i in clusters:
        for j in clusters:
            if i==j:
                pairwise_dists[i][j]=1
    return pairwise_dists