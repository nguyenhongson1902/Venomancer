import logging
import os
import time

import colorlog
import torch

from utils.parameters import Params


def record_time(params: Params, t=None, name=None):
    if t and name and params.save_timing == name or params.save_timing is True:
        torch.cuda.synchronize()
        params.timing_data[name].append(round(1000 * (time.perf_counter() - t)))

def create_table(params: dict):
    data = "| name | value | \n |-----|-----|"

    for key, value in params.items():
        data += '\n' + f"| {key} | {value} |"

    return data

def create_logger():
    """
        Setup the logging environment
    """
    log = logging.getLogger()  # root logger
    log.setLevel(logging.DEBUG)
    format_str = '%(asctime)s - %(levelname)-8s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    if os.isatty(2):
        cformat = '%(log_color)s' + format_str
        colors = {'DEBUG': 'reset',
                  'INFO': 'reset',
                  'WARNING': 'bold_yellow',
                  'ERROR': 'bold_red',
                  'CRITICAL': 'bold_red'}
        formatter = colorlog.ColoredFormatter(cformat, date_format,
                                              log_colors=colors)
    else:
        formatter = logging.Formatter(format_str, date_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    return logging.getLogger(__name__)

# FOR A3FL
def get_lr_a3fl(hlpr, epoch):
    '''epoch starts from 1'''
    lr_init = hlpr.params.lr_a3fl
    target_lr = hlpr.params.target_lr
    #if self.helper.config.dataset == 'cifar10':
    if epoch - 1 <= hlpr.params.epochs/2.:
        lr = (epoch - 1)*(target_lr - lr_init)/(hlpr.params.epochs/2.-1) + lr_init - (target_lr - lr_init)/(hlpr.params.epochs/2. - 1)
    else:
        lr = ((epoch - 1)-hlpr.params.epochs/2)*(-target_lr)/(hlpr.params.epochs/2) + target_lr

    if lr <= 0.002:
        lr = 0.002
    # else:
    #     raise NotImplementedError
    return lr

# FOR F3BA
# def handcraft(hlpr, task, local_model):
#     handcraft_rnd = 0
#     handcraft_rnd = handcraft_rnd + 1
#     if is_malicious and attacks.handcraft:
#         model = local_model
#         model.eval()
#         handcraft_loader, train_loader = handcraft_loader, train_loader

#         if attacks.previous_global_model is None:
#             attacks.previous_global_model = copy.deepcopy(model)
#             return
#         candidate_weights = attacks.search_candidate_weights(model, proportion=0.1)
#         attacks.previous_global_model = copy.deepcopy(model)

#         if attacks.params.handcraft_trigger:
#             print("Optimize Trigger:")
#             attacks.optimize_backdoor_trigger(model, candidate_weights, task, handcraft_loader)

#         print("Inject Candidate Filters:")
#         diff = attacks.inject_handcrafted_filters(model, candidate_weights, task, handcraft_loader)
#         if diff is not None and handcraft_rnd % 3 == 1:
#             print("Rnd {}: Inject Backdoor FC".format(handcraft_rnd))
#             attacks.inject_handcrafted_neurons(model, candidate_weights, task, diff, handcraft_loader)



