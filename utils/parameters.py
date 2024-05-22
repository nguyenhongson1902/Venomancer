from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict
import logging
import torch
from torch.utils.data import Subset
logger = logging.getLogger('logger')

@dataclass
class Params: 

    # Corresponds to the class module: tasks.mnist_task.MNISTTask
    # See other tasks in the task folder.
    task: str = 'MNIST'

    prefix: str = None # Prefix to the path to another place
    current_time: str = None
    exp: str = None
    name: str = None
    random_seed: int = 2025 # cifar-10, mnist, chestxray
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # training params
    start_epoch: int = 1
    epochs: int = None
    poison_epoch: int = None
    poison_epoch_stop: int = None
    log_interval: int = 1000

    # model arch is usually defined by the task
    resume_model: str = None
    lr: float = None
    decay: float = None
    momentum: float = None
    optimizer: str = None
    # data
    data_path: str = '.data/'
    batch_size: int = 64
    test_batch_size: int = 100
    # Do not apply transformations to the training images.
    transform_train: bool = True
    # For large datasets stop training earlier.
    max_batch_id: int = None
    # No need to set, updated by the Task class.
    input_shape = None # (channels, height, width)

    # attack params
    attack_type: str = 'venomancer'
    backdoor: bool = False
    backdoor_label: int = 8
    poisoning_proportion: float = 1.0  # backdoors proportion in backdoor loss
    synthesizer: str = 'pattern'
    backdoor_dynamic_position: bool = False

    # factors to balance losses
    fixed_scales: Dict[str, float] = None

    # optimizations:
    alternating_attack: float = None
    clip_batch: float = None
    # Disable BatchNorm and Dropout
    switch_to_eval: float = None

    # logging
    report_train_loss: bool = True
    log: bool = False
    save_model: bool = None
    save_on_epochs: List[int] = None
    save_scale_values: bool = False
    print_memory_consumption: bool = False
    save_timing: bool = False
    timing_data = None

    # Temporary storage for running values
    running_losses = None
    running_scales = None

    # FL params
    fl: bool = False
    fl_no_models: int = 100
    fl_local_epochs: int = 2
    fl_poison_epochs: int = None
    fl_total_participants: int = 200
    fl_eta: int = 1
    fl_sample_dirichlet: bool = False
    fl_dirichlet_alpha: float = None
    fl_diff_privacy: bool = False
    # FL attack details. Set no adversaries to perform the attack:
    fl_number_of_adversaries: int = 0
    fl_single_epoch_attack: int = None
    fl_weight_scale: int = 1

    attack: str = None #'ThrDFed' (3DFed), 'ModelRplace' (Model Replacement)
    
    #"Weak_DP", "Norm_Clipping", "Foolsgold", "FLAME", "RFLBAT", "Deepsight", "FLDetector"
    defense: str = None 
    lagrange_step: float = None
    random_neurons: List[int] = None
    noise_mask_alpha: float = None
    fl_adv_group_size: int = 0
    fl_num_neurons: int = 0

    # Marksman Settings
    clsmodel: str = None
    lr_atk: float = None
    eps: float = None
    # target_label: int = None # use backdoor_label instead
    num_classes: int = None # Depends on the dataset
    attack_portion: float = None # Default 1.0 (100%)
    alpha: float = None
    beta: float = None
    test_n_size: int = None # The number of examples to print out
    input_height: int = None
    input_width: int = None
    input_channel: int = None
    random_rotation: int = None # for post transformations
    random_crop: int = None # for post transformations
    dataset: str = None # 'mnist', 'cifar10', 'chestxray'
    mode: str = None # "all2one", "all2all"
    fixed_frequency: int = None
    gradmask_ratio: float = None
    # normal_training: bool = None
    # only_target_examples: bool = None

    fl_round_participants: List[int] = None
    fl_weight_contribution: Dict[int, float] = None
    fl_local_updated_models: Dict[int, Dict[str, torch.Tensor]] = None
    fl_number_of_samples_each_user: Dict[int, float] = None

    # current_acc_clean: float = 0.0
    # current_acc_poison: float = 0.0

    # local_backdoor_acc: float = None

    # multiplier: int = 3
    
    norm_bound: float = 3.0 # for Norm_Clipping defense
    stddev: float = 0.158 # for Weak_DP defense
    mode_krum: str = None # for Krum defense
    server_dataset: Subset = None # for distillation knowledge
    percentage_server_data: float = 0.0 # for distillation knowledge

    # FOR A3FL
    trigger_lr: float = None
    trigger_outter_epochs: int = None
    dm_adv_K: int = None
    dm_adv_model_count: int = None
    dm_adv_epochs: int = None
    noise_loss_lambda: int = None
    trigger_size: int = None
    attacker_retrain_times: int = None
    lr_a3fl: float = None
    target_lr: float = None
    retrain_times: int = None
    bkd_ratio: float = None

    # FOR F3BA
    

    def __post_init__(self):
        # enable logging anyways when saving statistics
        if self.save_model or self.save_timing or \
                self.print_memory_consumption:
            self.log = True

        if self.log:
            self.folder_path = self.prefix + f'saved_models/model_' \
                               f'{self.task}_{self.current_time}_{self.name}'

        self.running_losses = defaultdict(list)
        self.running_scales = defaultdict(list)
        self.timing_data = defaultdict(list)

    def to_dict(self):
        return asdict(self)