import torch
from attacks.attack import Attack
from attacks.loss_functions import compute_all_losses_and_grads


class Marksman(Attack):
    def __init__(self, params):
        self.params = params
        self.loss_tasks = ['normal', 'marksman']

    def perform_attack(self):
        # Inject a backdoor
        pass

