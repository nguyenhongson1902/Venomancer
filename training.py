import argparse
import shutil
from copy import deepcopy
from datetime import datetime

import numpy as np
import yaml
from prompt_toolkit import prompt
from tqdm import tqdm

from helper import Helper
from utils.utils import *
from utils.backdoor import IMAGENET_MIN, IMAGENET_MAX, PostTensorTransform, make_backdoor_batch, aggregate_atkmodels, pick_best_atkmodel, pick_backdoor_label_samples, get_grad_mask, apply_grad_mask, calculate_each_class_accuracy
from utils.utils import get_lr_a3fl

from attack_models.autoencoders import MNISTAutoencoder as Autoencoder
import pytorch_ssim

import pickle
import wandb
import copy


logger = logging.getLogger('logger')



# def train_normal_marksman(hlpr: HelperMarksman, epoch, model, optimizer, train_loader, attack=True, global_model=None):
#     criterion = hlpr.task.criterion
#     model.train()
#     # print(train_loader)
#     for i, data in tqdm(enumerate(train_loader), desc="Benign:"):
#         batch = hlpr.task.get_batch(i, data)
#         model.zero_grad()
#         loss = hlpr.attack.compute_loss(model, criterion, batch, attack, global_model)
#         loss.backward()
#         optimizer.step()
#         # Print the computed gradients
#         # for name, param in model.named_parameters():
#         #     print("DEBUG: ", name, param.grad)
#         #     break

#         # print("Benign loss: ", loss.item())
#         if i == hlpr.params.max_batch_id:
#             break
#     return

# def train_marksman(hlpr: HelperMarksman, epoch, model, optimizer, train_loader, atkmodel, tgtmodel, tgtoptimizer, 
#                     clip_image, post_transforms=None, target_transform=None, attack=True, global_model=None):
#     # # Get all examples in train_loader with target label and make a subset
#     # subset_targetlabel = []
#     # for image in train_loader.dataset:
#     #     if image[1] == hlpr.params.target_label[0]:
#     #         subset_targetlabel.append(image)
#     # # Make a dataloader with the subset
#     # trainloader_targetlabel = torch.utils.data.DataLoader(subset_targetlabel, batch_size=hlpr.params.batch_size, shuffle=True)
    
#     # Load poison_loader from disk
#     # Get the full path of the dataloaders folder
#     # import os
#     # import pickle
#     # dataloaders_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/LRBA/dataloaders'
#     # path = os.path.join(dataloaders_folder, f'poison_loader_{hlpr.params.current_time}.pkl')
#     # with open(path, 'rb') as f:
#     #     print("Loading poison_loader from disk...")
#     #     poison_loader = pickle.load(f)
    
#     criterion = hlpr.task.criterion
#     atkmodel.eval()
#     model.train()
#     tgtmodel.train()

#     # Training the atkmodel (generative model)
#     losslist = []
#     correct_transform = 0.0
#     train_loader_size = 0
#     # if hlpr.params.local_backdoor_acc < 0.85:
#     for i, data in tqdm(enumerate(train_loader), desc="Generative Model:"):
#     # for i, data in tqdm(enumerate(poison_loader)):
#         batch = hlpr.task.get_batch(i, data)
#         bs = batch.batch_size
#         batch_idx = i
#         train_loader_size += bs

#         ########################################
#         #### Update Trigger Function ####
#         ########################################
#         data, target = batch.inputs, batch.labels
#         # atktarget = sample_negative_labels(target, hlpr.params.num_classes).to(hlpr.params.device) #randomly sample any labels
#         atktarget = target_transform(target)
#         # noise = tgtmodel(data, atktarget) * hlpr.params.eps
#         #### Test on using only target examples to update tgtmodel ####
#         # for data_target, targetlabels in trainloader_targetlabel:
#         #     data_target, targetlabels = data_target.to(hlpr.params.device), targetlabels.to(hlpr.params.device)
#         #     noise = tgtmodel(data_target) * hlpr.params.eps
#         #     atkdata = clip_image(data_target + noise)
#         #     aug_atkdata = post_transforms(atkdata)
#         #     aug_data = post_transforms(data_target)

#         #     # Calculate loss
#         #     atkoutput = model(aug_atkdata)
#         #     loss_poison = criterion(atkoutput, targetlabels)
#         #     loss1 = loss_poison
#         #     losslist.append(loss_poison.item())

#         #     optimizer.zero_grad()
#         #     tgtoptimizer.zero_grad()
#         #     loss1.backward()
#         #     tgtoptimizer.step() #this is the slowest step

            
#         ################################

#         #### Attack on all examples ####
#         noise = tgtmodel(data) * hlpr.params.eps
#         atkdata = clip_image(data + noise)
        
#         aug_atkdata = post_transforms(atkdata)
#         aug_data = post_transforms(data)
#         ################################
#         # aug_data = post_transforms(data) # Good for training only on target samples
        
#         if hlpr.params.attack_portion < 1.0:
#             aug_atkdata = aug_atkdata[:int(hlpr.params.attack_portion*bs)]
#             atktarget = atktarget[:int(hlpr.params.attack_portion*bs)]

#         #### Attack on all examples ####
#         # Calculate loss
#         atkoutput = model(aug_atkdata)

#         atkpred = atkoutput.max(1, keepdim=True)[1]  # get the index of the max log-probability
#         correct_transform += atkpred.eq(
#                     atktarget.view_as(atkpred)).sum().item()

#         loss_poison = criterion(atkoutput, atktarget)
#         loss1 = loss_poison
#         losslist.append(loss_poison.item())
        
#         # optimizer.zero_grad()
#         model.zero_grad() # The same with optimizer.zero_grad() but it's safer
#         # tgtoptimizer.zero_grad()
#         tgtmodel.zero_grad() # The same with tgtoptimizer.zero_grad() but it's safer
#         loss1.backward()
#         tgtoptimizer.step() #this is the slowest step
#         ###############################

#         # print the loss
#         # if batch_idx % 10 == 0 or batch_idx == (len(train_loader)-1):
#         #     print('Train [{}] atkmodel Loss: CLS {:.4f}'.format(
#         #         epoch, loss1.item()))
#     # hlpr.params.local_backdoor_acc = correct_transform / train_loader_size
#     # print("Local Backdoor Accuracy: {:.4f}", hlpr.params.local_backdoor_acc)
#     # else:

#     # if correct_transform > 0.8:

#     # Backdoor injection
#     # for i, data in tqdm(enumerate(train_loader), desc="Backdoor Injection:"):
#     #     batch = hlpr.task.get_batch(i, data)
#     #     bs = batch.batch_size
#     #     batch_idx = i
#     #     data, target = batch.inputs, batch.labels
#     ###############################
#     #### Update the classifier ####
#     ###############################
#     # atktarget = sample_negative_labels(target, hlpr.params.num_classes).to(hlpr.params.device) #randomly sample any labels
#     atktarget = target_transform(target)
#     # noise = atkmodel(data, atktarget) * hlpr.params.eps
#     noise = atkmodel(data) * hlpr.params.eps
#     atkdata = clip_image(data + noise)
    
#     aug_atkdata = post_transforms(atkdata)
    
#     if hlpr.params.attack_portion < 1.0:
#         aug_atkdata = aug_atkdata[:int(hlpr.params.attack_portion*bs)]
#         atktarget = atktarget[:int(hlpr.params.attack_portion*bs)]
    
#     aug_data = post_transforms(data)

#     output = model(aug_data)
#     atkoutput = model(aug_atkdata)
#     loss_clean = criterion(output, target)
#     loss_poison = criterion(atkoutput, atktarget)
#     loss2 = loss_clean * hlpr.params.alpha + (1-hlpr.params.alpha) * loss_poison
#     # optimizer.zero_grad()
#     model.zero_grad() # The same with optimizer.zero_grad() but it's safer
#     loss2.backward()
#     optimizer.step()

#     # if batch_idx % 10 == 0 or batch_idx == (len(train_loader)-1):
#     if batch_idx % 10 == 0 or batch_idx == (len(train_loader)-1):
#         print('Train [{}] clsmodel Loss: clean {:.4f} poison {:.4f} ATK:{:.4f} CLS:{:.4f}'.format(
#             epoch, loss_clean.item(), loss_poison.item(), loss2.item(), loss1.item()))
                
#     atkloss = sum(losslist) / len(losslist)
#     return atkloss


# def sample_negative_labels(label, n_classes):
#     label_cpu = label.detach().cpu().numpy()
#     neg_label = [np.random.choice([e for e in range(n_classes) if e != l], 1)[0] for l in label_cpu]
#     neg_label = torch.tensor(np.array(neg_label))
#     return neg_label

# def run_fl_round_marksman(hlpr: HelperMarksman, epoch, target_transform=None):
#     global_model = hlpr.task.model
#     global_model.train()
#     local_model = hlpr.task.local_model
#     local_model.train()
#     round_participants = hlpr.task.sample_users_for_round(epoch)
#     weight_accumulator = hlpr.task.get_empty_accumulator()
#     # hlpr.params.fl_round_participants = [user.user_id for user in round_participants]

#     print(f"Round epoch {epoch} with participants: {[user.user_id for user in round_participants]} and weight: {hlpr.params.fl_weight_contribution}")
#     # log number of sample per user
#     print(f"Round epoch {epoch} with participants sample size: {[user.number_of_samples for user in round_participants]}")

#     atkmodel = None
#     clip_image = None
    
#     # Check if the folder exists
#     # lastest_atkmodel_folder = f'./saved_latest_atkmodel/{hlpr.params.current_time}'
#     # if not os.path.exists(lastest_atkmodel_folder):
#     #     os.makedirs(lastest_atkmodel_folder)
    
#     # models_to_average = [] # Averaging atkmodels from all malicious clients
#     # num_samples_list = [] # Contains the number of samples of each malicious client

#     for user in tqdm(round_participants, desc="Training Users:"):
#         hlpr.task.copy_params(global_model, local_model) # Copy global model to local model in terms of parameters
#         optimizer = hlpr.task.make_optimizer(local_model)
#         if user.compromised:
#             print("Training Malicious Client, user {}".format(user.user_id))
#             trainlosses = []
#             post_transforms = PostTensorTransform(hlpr.params).to(hlpr.params.device)
#             if 'mnist' in hlpr.params.task.lower():
#                 from attack_models.autoencoders import \
#                     MNISTAutoencoder as Autoencoder
#                 # from attack_models.unet import UNet
#                 # from backbones_unet.model.unet import Unet

#                 # if there exists a saved atkmodel, load it
#                 saved_atkmodel = f'./checkpoint/lira/lira_mnist_lenet_autoencoder_0.03.pt'
#                 if os.path.isfile(saved_atkmodel):
#                     # atkmodel = UNet(3).to(hlpr.params.device)
#                     # atkmodel = Unet(backbone="convnext_base", in_channels=3, num_classes=3).to(hlpr.params.device) # pretrained Unet
#                     # tgtmodel = Unet(backbone="convnext_base", in_channels=3, num_classes=3).to(hlpr.params.device) # pretrained Unet
                    
#                     atkmodel = Autoencoder().to(hlpr.params.device)

#                     print(f"TRAIN: Loading {saved_atkmodel}")

#                     with open(saved_atkmodel, 'rb') as f:
#                         atkmodel.load_state_dict(torch.load(saved_atkmodel, map_location=torch.device('cuda')))
#                         tgtmodel = deepcopy(atkmodel).to(hlpr.params.device)
#                         # tgtmodel.load_state_dict(save_dict['state_dict'])

#                     # with open(f"./saved_latest_atkmodel/{hlpr.params.current_time}/atkmodel_checkpoint_{user.user_id}.pkl", 'rb') as f:
#                     #     save_dict = pickle.load(f)
#                     #     atkmodel.load_state_dict(save_dict['state_dict'])
#                     #     tgtmodel.load_state_dict(save_dict['state_dict'])

#                     # atkmodel.load_state_dict(torch.load(f'./saved_latest_atkmodel/{hlpr.params.current_time}/atkmodel_weights_{user.user_id}.pth'))
                    
#                     # Copy of attack model
#                     # print(f"TRAIN: Loading ./saved_latest_atkmodel/{hlpr.params.current_time}/tgtmodel_weights_avg.pth")
#                     # print(f"TRAIN: Loading ./saved_latest_atkmodel/{hlpr.params.current_time}/atkmodel_weights_avg.pth")
#                     # tgtmodel = UNet(3).to(hlpr.params.device)

#                     # tgtmodel.load_state_dict(torch.load(f'./saved_latest_atkmodel/{hlpr.params.current_time}/atkmodel_weights_{user.user_id}.pth'))
#                 else: # Initialize a new atkmodel
#                     # atkmodel = UNet(3).to(hlpr.params.device)
#                     # atkmodel = Unet(backbone="convnext_base", in_channels=3, num_classes=3).to(hlpr.params.device) # pretrained Unet
#                     atkmodel = Autoencoder().to(hlpr.params.device)
                    
#                     # Copy of attack model
#                     # tgtmodel = UNet(3).to(hlpr.params.device)
#                     # tgtmodel = Unet(backbone="convnext_base", in_channels=3, num_classes=3).to(hlpr.params.device) # pretrained Unet
#                     tgtmodel = Autoencoder().to(hlpr.params.device)
#                 # atkmodel = Autoencoder().to(hlpr.params.device)
                
#                 # Copy of attack model
#                 # tgtmodel = Autoencoder().to(hlpr.params.device)
#                 clip_image = lambda x: torch.clamp(x, -1.0, 1.0) # clip_image must come with atkmodel
#             elif 'cifar10' in hlpr.params.task.lower():
#                 from attack_models.unet import UNet

#                 # if there exists a saved atkmodel, load it
#                 if os.path.isfile(f'./saved_latest_atkmodel/{hlpr.params.current_time}/atkmodel_weights.pth'):
#                     atkmodel = UNet(3).to(hlpr.params.device)
#                     print(f"TRAIN: Loading ./saved_latest_atkmodel/{hlpr.params.current_time}/atkmodel_weights.pth")
#                     atkmodel.load_state_dict(torch.load(f'./saved_latest_atkmodel/{hlpr.params.current_time}/atkmodel_weights.pth'))
                    
#                     # Copy of attack model
#                     tgtmodel = deepcopy(atkmodel).to(hlpr.params.device)
#                 else: # Initialize a new atkmodel
#                     atkmodel = UNet(3).to(hlpr.params.device)
                    
#                     # Copy of attack model
#                     tgtmodel = UNet(3).to(hlpr.params.device)
#                 clip_image = lambda x: torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
#             # Optimizer
#             # tgtoptimizer = torch.optim.Adam(tgtmodel.parameters(), lr=hlpr.params.lr_atk)
#             tgtoptimizer = torch.optim.SGD(tgtmodel.parameters(), lr=hlpr.params.lr_atk)
            
#             # if not user.user_id == 0:
#             #     continue
#             if user.user_id not in hlpr.task.adversaries:
#                 continue
            
#             for local_epoch in tqdm(range(hlpr.params.fl_poison_epochs)):
#                 trainloss = train_marksman(hlpr, local_epoch, local_model, optimizer,
#                         user.train_loader, atkmodel, tgtmodel, tgtoptimizer, clip_image, post_transforms, target_transform=target_transform,
#                         attack=True, global_model=global_model)
#                 trainlosses.append(trainloss)
#                 # print("trainlosses: ", trainlosses)
#                 atkmodel.train()
#                 tgtmodel.train()
#                 atkmodel.load_state_dict(tgtmodel.state_dict())
            

#             # Save the attack model weights and its number of samples using pickle
#             # save_dict = {'state_dict': atkmodel.state_dict(), 'num_samples': user.number_of_samples}
#             # Save save_dict using pickle
#             # with open(f'./saved_latest_atkmodel/{hlpr.params.current_time}/atkmodel_checkpoint_{user.user_id}.pkl', 'wb') as f:
#             #     pickle.dump(save_dict, f)


#             # models_to_average.append(atkmodel)
#             # num_samples_list.append(user.number_of_samples)


#             # Save the tgtmodel weights
#             # print(f"Saving ./saved_latest_atkmodel/{hlpr.params.current_time}/tgtmodel_weights_{user.user_id}.pth")
#             # torch.save(tgtmodel.state_dict(), f'./saved_latest_atkmodel/{hlpr.params.current_time}/tgtmodel_weights_{user.user_id}.pth')
            
#             # Save the atkmodel weights
#             # print(f"Saving ./saved_latest_atkmodel/{hlpr.params.current_time}/atkmodel_weights_{user.user_id}.pth")
#             # torch.save(atkmodel.state_dict(), f'./saved_latest_atkmodel/{hlpr.params.current_time}/atkmodel_weights_{user.user_id}.pth')
            

#         else:
#             # print('user.train_loader', user.train_loader) 
#             print("Training Benign Client, user {}".format(user.user_id))
#             for local_epoch in range(hlpr.params.fl_local_epochs):
#                 # hlpr.task.scheduler.step() # Update the learning rate at the beginning of each epoch
#                 train_normal_marksman(hlpr, local_epoch, local_model, optimizer,
#                         user.train_loader, attack=False)

#         local_update = hlpr.attack.get_fl_update(local_model, global_model)
#         hlpr.save_update(model=local_update, userID=user.user_id)
#         if user.compromised:
#             hlpr.attack.local_dataset = deepcopy(user.train_loader)


#     # if models_to_average:
#     #     from backbones_unet.model.unet import Unet
#     #     average_model = Unet(backbone="convnext_base", in_channels=3, num_classes=3).to(hlpr.params.device)
#     #     for param in average_model.parameters():
#     #         param.data.zero_()  # Initialize the parameters with zeros
#     #     for i, model in enumerate(models_to_average):
#     #         for param_avg, param_model in zip(average_model.parameters(), model.parameters()):
#     #             param_avg.data += param_model.data * (num_samples_list[i] / sum(num_samples_list))
#     #     print(f"Saving ./saved_latest_atkmodel/{hlpr.params.current_time}/atkmodel_weights_avg.pth")
#     #     torch.save(average_model.state_dict(), f'./saved_latest_atkmodel/{hlpr.params.current_time}/atkmodel_weights_avg.pth')


#     # hlpr.attack.perform_attack(global_model, epoch)
#     hlpr.defense.aggr(weight_accumulator, global_model)
#     hlpr.task.update_global_model(weight_accumulator, global_model)
    
#     return atkmodel, clip_image


# def run_marksman(hlpr: HelperMarksman, target_transform=None):
#     for epoch in tqdm(range(hlpr.params.start_epoch,
#                        hlpr.params.epochs + 1), desc="Marksman Attack"):
#         atkmodel, clip_image = run_fl_round_marksman(hlpr, epoch, target_transform=target_transform)
        
#         # from attack_models.unet import UNet
#         # from backbones_unet.model.unet import Unet

#         # models_to_average = []
#         # num_samples_list = []
#         # # average_model = UNet(3).to(hlpr.params.device)
#         # average_model = Unet(backbone="convnext_base", in_channels=3, num_classes=3).to(hlpr.params.device)
#         # for param in average_model.parameters():
#         #     param.data.zero_()  # Initialize the parameters with zeros

#         # files = os.listdir(f"./saved_latest_atkmodel/{hlpr.params.current_time}")
#         # if len(files) > 0:
#         #     num_models = len(files)
#         #     print("Averaging atkmodels")
#         #     for file in files:
#         #         # atkmodel = UNet(3).to(hlpr.params.device)
#         #         atkmodel = Unet(backbone="convnext_base", in_channels=3, num_classes=3).to(hlpr.params.device)
#         #         if "avg" not in file:
#         #             print(f"Loading ./saved_latest_atkmodel/{hlpr.params.current_time}/" + file)
#         #             with open(f"./saved_latest_atkmodel/{hlpr.params.current_time}/" + file, 'rb') as f:
#         #                 save_dict = pickle.load(f)
#         #                 atkmodel.load_state_dict(save_dict['state_dict'])
#         #                 num_samples = save_dict['num_samples']

#         #             # atkmodel.load_state_dict(torch.load(f'./saved_latest_atkmodel/{hlpr.params.current_time}/' + file))
#         #             models_to_average.append(atkmodel)
#         #             num_samples_list.append(num_samples)

#         #     if models_to_average:
#         #         for i, model in enumerate(models_to_average):
#         #             for param_avg, param_model in zip(average_model.parameters(), model.parameters()):
#         #                 param_avg.data += param_model.data * (num_samples_list[i] / sum(num_samples_list))
                
#         #         for param_avg in average_model.parameters():
#         #             param_avg.data /= num_models

#         #         print(f"Saving ./saved_latest_atkmodel/{hlpr.params.current_time}/atkmodel_weights_avg.pth")
#         #         torch.save(average_model.state_dict(), f'./saved_latest_atkmodel/{hlpr.params.current_time}/atkmodel_weights_avg.pth')
        
#         acc_clean, acc_poison = test_marksman(hlpr, epoch, atkmodel, hlpr.task.model, 
#                                 clip_image, target_transform, post_transforms=None)
#         hlpr.params.current_acc_clean = acc_clean
#         hlpr.params.current_acc_poison = acc_poison
#         # metric = test(hlpr, epoch, backdoor=False)
#         # hlpr.record_accuracy(metric, test(hlpr, epoch, backdoor=True), epoch)

#         # hlpr.save_model(hlpr.task.model, epoch, metric)



# def test_marksman(hlpr: HelperMarksman, epoch, atkmodel, globalmodel, 
#                     clip_image, target_transform, post_transforms=None):
#     test_loader = hlpr.task.test_loader
#     criterion = hlpr.task.criterion
#     globalmodel.eval()

#     # saved_atkmodel = f'./checkpoint/lira/lira_mnist_lenet_autoencoder_0.03.pt'

#     # if os.path.isfile(f"./saved_latest_atkmodel/{hlpr.params.current_time}/atkmodel_weights_avg.pth"):
#     # if not os.path.isfile(saved_atkmodel):
#     if atkmodel and clip_image:
#         print("Using the latest atkmodel")
#         # from attack_models.unet import UNet
#         # from backbones_unet.model.unet import Unet

#         # # average_model = UNet(3).to(hlpr.params.device)
#         # average_model = Unet(backbone="convnext_base", in_channels=3, num_classes=3).to(hlpr.params.device)
#         # print(f"TEST: Loading ./saved_latest_atkmodel/{hlpr.params.current_time}/atkmodel_weights_avg.pth")
#         # average_model.load_state_dict(torch.load(f"./saved_latest_atkmodel/{hlpr.params.current_time}/atkmodel_weights_avg.pth"))
#         # atkmodel = average_model
#     # if atkmodel:
#         # assert clip_image is not None, "clip_image is None"
#         atkmodel.eval()

#         if 'mnist' in hlpr.params.task.lower():
#             clip_image = lambda x: torch.clamp(x, -1.0, 1.0)
#         elif 'cifar10' in hlpr.params.task.lower():
#             clip_image = lambda x: torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)

#         correct = 0
#         correct_transform = 0
#         test_loss = 0
#         test_transform_loss = 0

#         with torch.no_grad():
#             for i, data in tqdm(enumerate(test_loader), desc="Test:"):
#                 batch = hlpr.task.get_batch(i, data)
#                 bs = batch.batch_size
#                 batch_idx = i

#                 data, target = batch.inputs, batch.labels
#                 output = globalmodel(data)
                
#                 test_loss += criterion(output, target).item() * bs  # sum up batch loss
#                 pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
#                 correct += pred.eq(target.view_as(pred)).sum().item()

#                 #noise = atkmodel(data) * args.test_eps
#                 #atkdata = clip_image(data + noise)
#                 #atkoutput = scratchmodel(atkdata)
                
#                 # atktarget = sample_negative_labels(target, hlpr.params.num_classes).to(hlpr.params.device) #randomly sample any labels
#                 atktarget = target_transform(target)
#                 # noise = atkmodel(data, atktarget) * hlpr.params.eps
#                 noise = atkmodel(data) * hlpr.params.eps
#                 atkdata = clip_image(data + noise * hlpr.params.multiplier)
#                 atkoutput = globalmodel(atkdata)
                
#                 test_transform_loss += criterion(atkoutput, atktarget).item() * bs  # sum up batch loss
#                 atkpred = atkoutput.max(1, keepdim=True)[1]  # get the index of the max log-probability
#                 correct_transform += atkpred.eq(
#                     atktarget.view_as(atkpred)).sum().item()
                

#         test_loss /= len(test_loader.dataset)
#         test_transform_loss /= len(test_loader.dataset)

#         correct /= len(test_loader.dataset)
#         correct_transform /= len(test_loader.dataset)

#         print(
#             '\nTest set: Loss: clean {:.4f} poison {:.4f}, Accuracy: clean {:.4f} poison {:.4f}'.format(
#                 test_loss, test_transform_loss,
#                 correct, correct_transform
#             ))
            
#         print(target[:hlpr.params.test_n_size].cpu().numpy())
#         print(atktarget[:hlpr.params.test_n_size].cpu().numpy())
#         print(atkpred[:hlpr.params.test_n_size].view(-1).cpu().numpy())
#         return correct, correct_transform
#     else:
#         assert atkmodel is None, "atkmodel is not None"
#         assert clip_image is None, "clip_image is not None"
        
        
#         # Check if the model file exists or not
#         # if os.path.isfile(f'./saved_latest_atkmodel/{hlpr.params.current_time}/atkmodel_weights_avg.pth'):
#         #     # Load the saved latest atkmodel weights in saved_latest_atkmodel folder, given the atkmodel is UNet
#         #     from attack_models.unet import UNet
#         #     from backbones_unet.model.unet import Unet

#         #     # average_model = UNet(3).to(hlpr.params.device)
#         #     average_model = Unet(backbone="convnext_base", in_channels=3, num_classes=3).to(hlpr.params.device)
#         #     print(f"TEST: Loading ./saved_latest_atkmodel/{hlpr.params.current_time}/atkmodel_weights_avg.pth")
#         #     atkmodel = average_model
#         #     atkmodel.load_state_dict(torch.load(f'./saved_latest_atkmodel/{hlpr.params.current_time}/atkmodel_weights_avg.pth'))
#         #     atkmodel.eval()

#         saved_atkmodel = f'./checkpoint/lira/lira_mnist_lenet_autoencoder_0.03.pt'

#         atkmodel = Autoencoder().to(hlpr.params.device)
#         print("Loading atkmodel...")
#         # Loading with map_location
#         atkmodel.load_state_dict(torch.load(saved_atkmodel, map_location=torch.device('cuda')))
#         # atkmodel.load_state_dict(torch.load(saved_atkmodel))
        
#         if 'mnist' in hlpr.params.task.lower():
#             clip_image = lambda x: torch.clamp(x, -1.0, 1.0)
#         elif 'cifar10' in hlpr.params.task.lower():
#             clip_image = lambda x: torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
        
#         correct = 0    
#         correct_transform = 0
#         test_loss = 0
#         test_transform_loss = 0

#         with torch.no_grad():
#             for i, data in tqdm(enumerate(test_loader), desc="Test:"):
#                 batch = hlpr.task.get_batch(i, data)
#                 bs = batch.batch_size
#                 batch_idx = i

#                 data, target = batch.inputs, batch.labels
#                 output = globalmodel(data)
#                 test_loss += criterion(output, target).item() * bs  # sum up batch loss
#                 pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
#                 correct += pred.eq(target.view_as(pred)).sum().item()

#                 atktarget = target_transform(target)
#                 if atkmodel:
#                     noise = atkmodel(data) * hlpr.params.eps
#                     atkdata = clip_image(data + noise * hlpr.params.multiplier)
#                 else:
#                     atkdata = data

#                 atkoutput = globalmodel(atkdata)

#                 test_transform_loss += criterion(atkoutput, atktarget).item() * bs  # sum up batch loss
#                 atkpred = atkoutput.max(1, keepdim=True)[1]  # get the index of the max log-probability
#                 correct_transform += atkpred.eq(
#                     atktarget.view_as(atkpred)).sum().item()
                

#         test_loss /= len(test_loader.dataset)
#         test_transform_loss /= len(test_loader.dataset)

#         correct /= len(test_loader.dataset)
#         correct_transform /= len(test_loader.dataset)

#         print(
#             '\nTest set: Loss: clean {:.4f} poison {:.4f}, Accuracy: clean {:.4f} poison {:.4f}'.format(
#                 test_loss, test_transform_loss,
#                 correct, correct_transform
#             ))
            
#         return correct, correct_transform

# def train(hlpr: Helper, epoch, model, optimizer, train_loader, attack=True, global_model=None):
#     criterion = hlpr.task.criterion
#     model.train()
#     for i, data in tqdm(enumerate(train_loader)):
#         batch = hlpr.task.get_batch(i, data)
#         model.zero_grad()
#         loss = hlpr.attack.compute_blind_loss(model, criterion, batch, attack, global_model)
#         loss.backward()
#         optimizer.step()

#         if i == hlpr.params.max_batch_id:
#             break
#     return
def train_with_noise_patch(hlpr: Helper, local_epoch, local_model, local_optimizer, local_train_loader, attack=True, global_model=None, 
          atkmodel=None, tgtmodel=None, tgtoptimizer=None, target_transform=None, clip_image=None, post_transforms=None):
    if attack:
        atkmodel.eval()
        local_model.train()
        tgtmodel.train()

        atklosslist = []
        cleanlosslist = []
        backdoorlosslist = []
        local_dataset_size = 0
        for batch_idx, data_labels in tqdm(enumerate(local_train_loader), total=len(local_train_loader)):
            batch = hlpr.task.get_batch(batch_idx, data_labels)
            bs = batch.batch_size
            local_dataset_size += bs

            data, target = batch.inputs, batch.labels
            atktarget = target_transform(target) # Flipping label

            # First, update the atkmodel weights using tgtoptimizer.step(), fix local_model weights
            noise = tgtmodel(data) * hlpr.params.eps # Do I need to pass atktarget?
            atkdata = clip_image(data + noise)

            augmented_atkdata = post_transforms(atkdata)
            augmented_data = post_transforms(data)

            # Calculus loss
            atkoutput = local_model(augmented_atkdata)
            atkloss = hlpr.task.criterion(atkoutput, atktarget)
            atklosslist.append(sum(atkloss))

            # local_optimizer.zero_grad()
            tgtoptimizer.zero_grad()
            atkloss.mean().backward(retain_graph=True)
            tgtoptimizer.step()

            # Second, update local_model weights using local_optimizer.step(), fix atkmodel weights
            backdoored_batch = hlpr.attack.synthesizer.make_backdoor_batch(batch, test=True, attack=True)
            atkdata = backdoored_batch.inputs
            atktarget = backdoored_batch.labels

            augmented_atkdata = post_transforms(atkdata)
            augmented_data = post_transforms(data)

            output = local_model(augmented_data)
            atkoutput = local_model(augmented_atkdata)
            clean_loss = hlpr.task.criterion(output, target)
            cleanlosslist.append(sum(clean_loss))
            backdoor_loss = hlpr.task.criterion(atkoutput, atktarget)
            backdoorlosslist.append(sum(backdoor_loss))
            total_loss = hlpr.params.alpha * clean_loss + (1-hlpr.params.alpha) * backdoor_loss

            local_optimizer.zero_grad()
            total_loss.mean().backward()
            local_optimizer.step()

            if batch_idx % 10 or batch_idx == len(local_train_loader) - 1:
                print(f"Train Malicious [Batch {batch_idx}/{len(local_train_loader)}, Epoch {local_epoch}], Generative Loss {atkloss.mean().item():.4f}, Classifier: Clean Loss {clean_loss.mean().item():.4f} " \
                    f"Backdoor Loss {backdoor_loss.mean().item():.4f} Total {total_loss.mean().item():.4f}")
        
        atkloss = sum(atklosslist) / local_dataset_size
        cleanloss = sum(cleanlosslist) / local_dataset_size
        backdoorloss = sum(backdoorlosslist) / local_dataset_size

        return atkloss, cleanloss, backdoorloss
    else:
        cleanlosslist = []
        local_dataset_size = 0
        for batch_idx, data_labels in tqdm(enumerate(local_train_loader), total=len(local_train_loader)):
            batch = hlpr.task.get_batch(batch_idx, data_labels)
            bs = batch.batch_size
            local_dataset_size += bs

            data, target = batch.inputs, batch.labels
            augmented_data = post_transforms(data)

            output = local_model(augmented_data)
            loss = hlpr.task.criterion(output, target)
            cleanlosslist.append(sum(loss))

            loss.mean().backward()
            local_optimizer.step()

            if batch_idx % 10 or batch_idx == len(local_train_loader) - 1:
                print(f"Train Benign [Batch {batch_idx}/{len(local_train_loader)}, Epoch {local_epoch}], Classifier: Clean Loss {loss.mean().item():.4f}")
        cleanloss = sum(cleanlosslist) / local_dataset_size
        return cleanloss

def train_like_a_gan_clean_label(hlpr: Helper, local_epoch, local_model, local_optimizer, local_train_loader, attack=True, global_model=None, 
          atkmodel=None, tgtmodel=None, tgtoptimizer=None, target_transform=None, clip_image=None, post_transforms=None):
    if attack:
        for batch_idx, data_labels in tqdm(enumerate(local_train_loader), total=len(local_train_loader)):
            batch = hlpr.task.get_batch(batch_idx, data_labels)
            data, target = batch.inputs, batch.labels
            

        # atkmodel.eval()
        local_model.train()
        tgtmodel.train()

        atklosslist = []
        cleanlosslist = []
        backdoorlosslist = []
        local_dataset_size = 0
        for batch_idx, data_labels in tqdm(enumerate(local_train_loader), total=len(local_train_loader)):
            batch = hlpr.task.get_batch(batch_idx, data_labels)
            bs = batch.batch_size
            local_dataset_size += bs

            data, target = batch.inputs, batch.labels

            atktarget = target_transform(target) # Flipping label
            backdoor_label_data, backdoor_label_target = pick_backdoor_label_samples(hlpr, data, target)
            # First, update the atkmodel weights using tgtoptimizer.step(), fix local_model weights
            # noise = tgtmodel(data) * hlpr.params.eps # Do I need to pass atktarget?
            if len(backdoor_label_target) > 0:
                noise = tgtmodel(backdoor_label_data) * hlpr.params.eps
                atkdata = clip_image(backdoor_label_data + noise)

                augmented_atkdata = post_transforms(atkdata)
                augmented_data = post_transforms(data)

                # Calculus loss
                atkoutput = local_model(augmented_atkdata)
                atkloss = hlpr.task.criterion(atkoutput, backdoor_label_target)
                atklosslist.append(sum(atkloss))

                # local_optimizer.zero_grad()
                tgtoptimizer.zero_grad()
                atkloss.mean().backward(retain_graph=True)
                tgtoptimizer.step() # Only update the weights of the generative model

            # Second, update local_model weights using local_optimizer.step(), fix atkmodel weights
            # noise = tgtmodel(data) * hlpr.params.eps
            noise = tgtmodel(data) * hlpr.params.eps
            atkdata = clip_image(data + noise)

            augmented_atkdata = post_transforms(atkdata)
            augmented_data = post_transforms(data)

            output = local_model(augmented_data)
            atkoutput = local_model(augmented_atkdata.detach())
            clean_loss = hlpr.task.criterion(output, target)
            cleanlosslist.append(sum(clean_loss))
            backdoor_loss = hlpr.task.criterion(atkoutput, atktarget)
            backdoorlosslist.append(sum(backdoor_loss))
            total_loss = hlpr.params.alpha * clean_loss + (1-hlpr.params.alpha) * backdoor_loss

            local_optimizer.zero_grad()
            total_loss.mean().backward()
            local_optimizer.step() # Only update the weights of the classifier model

            if len(backdoor_label_target) > 0:
                if batch_idx % 10 or batch_idx == len(local_train_loader) - 1:
                    print(f"Train Malicious [Batch {batch_idx}/{len(local_train_loader)}, Epoch {local_epoch}], Generative Loss {atkloss.mean().item():.4f}, Classifier: Clean Loss {clean_loss.mean().item():.4f} " \
                        f"Backdoor Loss {backdoor_loss.mean().item():.4f} Total {total_loss.mean().item():.4f}")
                atkloss = sum(atklosslist) / local_dataset_size
            else:
                if batch_idx % 10 or batch_idx == len(local_train_loader) - 1:
                    print(f"Train Malicious [Batch {batch_idx}/{len(local_train_loader)}, Epoch {local_epoch}], Classifier: Clean Loss {clean_loss.mean().item():.4f} " \
                        f"Backdoor Loss {backdoor_loss.mean().item():.4f} Total {total_loss.mean().item():.4f}")
                atkloss = None
        
        cleanloss = sum(cleanlosslist) / local_dataset_size
        backdoorloss = sum(backdoorlosslist) / local_dataset_size

        return atkloss, cleanloss, backdoorloss
    else:
        cleanlosslist = []
        local_dataset_size = 0
        for batch_idx, data_labels in tqdm(enumerate(local_train_loader), total=len(local_train_loader)):
            batch = hlpr.task.get_batch(batch_idx, data_labels)
            bs = batch.batch_size
            local_dataset_size += bs

            data, target = batch.inputs, batch.labels
            augmented_data = post_transforms(data)

            output = local_model(augmented_data)
            loss = hlpr.task.criterion(output, target)
            cleanlosslist.append(sum(loss))

            loss.mean().backward()
            local_optimizer.step()

            if batch_idx % 10 or batch_idx == len(local_train_loader) - 1:
                print(f"Train Benign [Batch {batch_idx}/{len(local_train_loader)}, Epoch {local_epoch}], Classifier: Clean Loss {loss.mean().item():.4f}")
        cleanloss = sum(cleanlosslist) / local_dataset_size
        return cleanloss

def train_like_a_gan_learnable_eps(hlpr: Helper, local_epoch, local_model, local_optimizer, local_train_loader, attack=True, global_model=None, 
          atkmodel=None, tgtmodel=None, tgtoptimizer=None, target_transform=None, clip_image=None, post_transforms=None, mask_grad_list=None):
    if attack:
        # atkmodel.eval()
        local_model.train()
        tgtmodel.train()

        learnable_eps = torch.tensor(hlpr.params.eps, requires_grad=True)
        learnable_eps = max(torch.tensor(0.05, requires_grad=False), learnable_eps)
        optimizer_eps = torch.optim.Adam([learnable_eps], lr=3*1e-6)

        atklosslist = []
        cleanlosslist = []
        backdoorlosslist = []
        local_dataset_size = 0
        for batch_idx, data_labels in tqdm(enumerate(local_train_loader), total=len(local_train_loader)):
            batch = hlpr.task.get_batch(batch_idx, data_labels)
            bs = batch.batch_size
            local_dataset_size += bs

            data, target = batch.inputs, batch.labels

            # atktarget = target_transform(target) # Flipping label
            atktarget = target_transform(target, n_classes=10)

            # First, update the atkmodel weights using tgtoptimizer.step(), fix local_model weights
            # noise = tgtmodel(data) * hlpr.params.eps # Do I need to pass atktarget?
            noise = tgtmodel(data, atktarget) * learnable_eps
            # noise = tgtmodel(data, atktarget)
            # noise = torch.clamp(noise, -hlpr.params.eps, hlpr.params.eps)

            atkdata = clip_image(data + noise)

            augmented_atkdata = post_transforms(atkdata)
            augmented_data = post_transforms(data)

            # Calculus loss
            atkoutput = local_model(augmented_atkdata)
            atkloss = hlpr.task.criterion(atkoutput, atktarget)
            atklosslist.append(sum(atkloss))

            # visual_loss = torch.sum(torch.square(atkdata - data))


            # local_optimizer.zero_grad()
            tgtoptimizer.zero_grad()
            atkloss.mean().backward(retain_graph=True)
            # (atkloss + visual_loss).mean().backward(retain_graph=True)
            tgtoptimizer.step() # Only update the weights of the generative model

            visual_loss = torch.sum(torch.square(atkdata - data)) / bs
            optimizer_eps.zero_grad()
            visual_loss.backward()
            optimizer_eps.step()
            hlpr.params.eps = learnable_eps.item()
            print("eps", hlpr.params.eps)

            # Second, update local_model weights using local_optimizer.step(), fix atkmodel weights
            # noise = tgtmodel(data) * hlpr.params.eps
            noise = tgtmodel(data, atktarget) * hlpr.params.eps
            # noise = tgtmodel(data, atktarget)
            # noise = torch.clamp(noise, -hlpr.params.eps, hlpr.params.eps)
            atkdata = clip_image(data + noise)

            augmented_atkdata = post_transforms(atkdata)
            augmented_data = post_transforms(data)

            output = local_model(augmented_data)
            atkoutput = local_model(augmented_atkdata.detach())
            clean_loss = hlpr.task.criterion(output, target)
            cleanlosslist.append(sum(clean_loss))
            backdoor_loss = hlpr.task.criterion(atkoutput, atktarget)
            backdoorlosslist.append(sum(backdoor_loss))
            total_loss = hlpr.params.alpha * clean_loss + (1-hlpr.params.alpha) * backdoor_loss
            local_optimizer.zero_grad()
            total_loss.mean().backward()

            # Apply grad_mask before updating weights
            print(f"Apply grad mask, ratio {hlpr.params.gradmask_ratio}")
            apply_grad_mask(local_model, mask_grad_list)

            local_optimizer.step() # Only update the weights of the classifier model

            if batch_idx % 10 or batch_idx == len(local_train_loader) - 1:
                print(f"Train Malicious [Batch {batch_idx}/{len(local_train_loader)}, Epoch {local_epoch}], Generative Loss {atkloss.mean().item():.4f},  Visual Loss: {visual_loss.item():.4f}, Classifier: Clean Loss {clean_loss.mean().item():.4f} " \
                    f"Backdoor Loss {backdoor_loss.mean().item():.4f} Total {total_loss.mean().item():.4f}")
        
        atkloss = sum(atklosslist) / local_dataset_size
        cleanloss = sum(cleanlosslist) / local_dataset_size
        backdoorloss = sum(backdoorlosslist) / local_dataset_size

        return atkloss, cleanloss, backdoorloss
    else:
        cleanlosslist = []
        local_dataset_size = 0
        for batch_idx, data_labels in tqdm(enumerate(local_train_loader), total=len(local_train_loader)):
            batch = hlpr.task.get_batch(batch_idx, data_labels)
            bs = batch.batch_size
            local_dataset_size += bs

            data, target = batch.inputs, batch.labels
            augmented_data = post_transforms(data)

            output = local_model(augmented_data)
            loss = hlpr.task.criterion(output, target)
            cleanlosslist.append(sum(loss))

            loss.mean().backward()
            local_optimizer.step()

            if batch_idx % 10 or batch_idx == len(local_train_loader) - 1:
                print(f"Train Benign [Batch {batch_idx}/{len(local_train_loader)}, Epoch {local_epoch}], Classifier: Clean Loss {loss.mean().item():.4f}")
        cleanloss = sum(cleanlosslist) / local_dataset_size
        return cleanloss

def train_like_a_gan_with_visual_loss(hlpr: Helper, local_epoch, local_model, local_optimizer, local_train_loader, attack=True, global_model=None, 
                                    atkmodel=None, tgtmodel=None, tgtoptimizer=None, target_transform=None, clip_image=None, post_transforms=None):
    if attack:
        # atkmodel.eval()
        local_model.train()
        tgtmodel.train()

        atklosslist = []
        cleanlosslist = []
        backdoorlosslist = []
        local_dataset_size = 0
        for batch_idx, data_labels in tqdm(enumerate(local_train_loader), total=len(local_train_loader)):
            batch = hlpr.task.get_batch(batch_idx, data_labels)
            bs = batch.batch_size
            local_dataset_size += bs

            data, target = batch.inputs, batch.labels

            # atktarget = target_transform(target) # Flipping label
            atktarget = target_transform(target, n_classes=hlpr.params.num_classes)

            # First, update the atkmodel weights using tgtoptimizer.step(), fix local_model weights
            # noise = tgtmodel(data) * hlpr.params.eps # Do I need to pass atktarget?
            noise = tgtmodel(data, atktarget)
            # print("data+noise min", (data + noise).min())
            # print("data+noise max", (data + noise).max())
            atkdata = clip_image(data + noise)
            # atkdata = data + noise

            # augmented_atkdata = post_transforms(atkdata)
            # augmented_data = post_transforms(data)
            augmented_atkdata = atkdata.clone()
            augmented_data = data.clone()

            # Calculus loss
            atkoutput = local_model(augmented_atkdata)
            # atkoutput = atkoutput.logits # test with microsoft/resnet-50, remove this line for other models
            atkloss = hlpr.task.criterion(atkoutput, atktarget)
            atklosslist.append(sum(atkloss))

            # local_optimizer.zero_grad()
            tgtoptimizer.zero_grad() # worked well, Feb 4, 2023
            # tgtoptimizer.zero_grad(set_to_none=True)
            # visual_loss = torch.sum(torch.square(atkdata - data), dim=(1, 2, 3))
            # (0.999*atkloss + 0.001*visual_loss).mean().backward(retain_graph=True)
            # (0.9999*atkloss + 0.0001*visual_loss).mean().backward(retain_graph=True) # exp 85
            # (atkloss + visual_loss).mean().backward(retain_graph=True)
            # (0.9*atkloss + 0.1*visual_loss).mean().backward(retain_graph=True) # quite good
            # (0.5*atkloss + 0.5*visual_loss).mean().backward(retain_graph=True) # not good
            # (0.8*atkloss + 0.2*visual_loss).mean().backward(retain_graph=True)

            visual_loss = 1 - torch.nn.functional.cosine_similarity(atkdata.flatten(start_dim=1), data.flatten(start_dim=1)) # Comment out to test with other visual loss, this works
            # visual_loss = torch.nn.functional.mse_loss(atkdata, data, reduction="none").flatten(start_dim=1).sum(axis=1)
            # (atkloss + visual_loss).mean().backward(retain_graph=True)
            # (0.5*atkloss + 0.5*visual_loss).mean().backward(retain_graph=True)
            # (0.9*atkloss + 0.1*visual_loss).mean().backward(retain_graph=True)
            # (hlpr.params.beta*atkloss + (1 - hlpr.params.beta)*visual_loss).mean().backward(retain_graph=True)
            (hlpr.params.beta*atkloss + (1 - hlpr.params.beta)*visual_loss).mean().backward() # Comment out to test with other visual loss, this works
            # (hlpr.params.beta*atkloss + (1 - hlpr.params.beta)*visual_loss).mean().backward()
            # (0.5*atkloss + 0.5*visual_loss).mean().backward(retain_graph=True)

            # ssim = pytorch_ssim.SSIM(window_size=11)
            # visual_loss = (ssim(atkdata, data) + 1) / 2
            # (atkloss + visual_loss).mean().backward(retain_graph=True)

            # huber_loss = torch.nn.HuberLoss(reduction='none', delta=1.0)
            # visual_loss = huber_loss(atkdata, data).sum(dim=(1,2,3))
            # # (atkloss + visual_loss).mean().backward(retain_graph=True)
            # (0.9*atkloss + 0.1*visual_loss).mean().backward(retain_graph=True)
            
            tgtoptimizer.step() # Only update the weights of the generative model

            

            # Second, update local_model weights using local_optimizer.step(), fix atkmodel weights
            # noise = tgtmodel(data) * hlpr.params.eps
            noise = tgtmodel(data, atktarget)
            atkdata = clip_image(data + noise)
            # atkdata = data + noise

            # augmented_atkdata = post_transforms(atkdata)
            # augmented_data = post_transforms(data)
            augmented_atkdata = atkdata.clone()
            augmented_data = data.clone()

            output = local_model(augmented_data)
            # output = output.logits # test with microsoft/resnet-50, remove this line for other models
            atkoutput = local_model(augmented_atkdata.detach())
            # atkoutput = atkoutput.logits # test with microsoft/resnet-50, remove this line for other models
            clean_loss = hlpr.task.criterion(output, target)
            cleanlosslist.append(sum(clean_loss))
            backdoor_loss = hlpr.task.criterion(atkoutput, atktarget)
            backdoorlosslist.append(sum(backdoor_loss))
            total_loss = hlpr.params.alpha * clean_loss + (1-hlpr.params.alpha) * backdoor_loss

            local_optimizer.zero_grad() # worked well, Feb 4, 2023
            # local_optimizer.zero_grad(set_to_none=True)
            total_loss.mean().backward()
            local_optimizer.step() # Only update the weights of the classifier model

            if batch_idx % 10 or batch_idx == len(local_train_loader) - 1:
                print(f"Train Malicious [Batch {batch_idx}/{len(local_train_loader)}, Epoch {local_epoch}], Generative Loss {atkloss.mean().item():.4f}, Visual Loss: {visual_loss.mean().item():.4f}, Classifier: Clean Loss {clean_loss.mean().item():.4f} " \
                    f"Backdoor Loss {backdoor_loss.mean().item():.4f} Total {total_loss.mean().item():.4f}")
        
        atkloss = sum(atklosslist) / local_dataset_size
        cleanloss = sum(cleanlosslist) / local_dataset_size
        backdoorloss = sum(backdoorlosslist) / local_dataset_size

        return atkloss, cleanloss, backdoorloss
    else:
        cleanlosslist = []
        local_dataset_size = 0
        for batch_idx, data_labels in tqdm(enumerate(local_train_loader), total=len(local_train_loader)):
            batch = hlpr.task.get_batch(batch_idx, data_labels)
            bs = batch.batch_size
            local_dataset_size += bs

            data, target = batch.inputs, batch.labels
            # augmented_data = post_transforms(data)
            augmented_data = data.clone()

            output = local_model(augmented_data)
            loss = hlpr.task.criterion(output, target)
            cleanlosslist.append(sum(loss))

            loss.mean().backward()
            local_optimizer.step()

            if batch_idx % 10 or batch_idx == len(local_train_loader) - 1:
                print(f"Train Benign [Batch {batch_idx}/{len(local_train_loader)}, Epoch {local_epoch}], Classifier: Clean Loss {loss.mean().item():.4f}")
        cleanloss = sum(cleanlosslist) / local_dataset_size
        return cleanloss
    
def train_like_a_gan_with_visual_loss_check_durability(hlpr: Helper, local_epoch, local_model, local_optimizer, local_train_loader, attack=True, global_model=None, 
                                    atkmodel=None, tgtmodel=None, tgtoptimizer=None, target_transform=None, clip_image=None, post_transforms=None, mask_grad_list=None):
    if attack:
        # atkmodel.eval() 3))
            # (0.999*atkloss + 0.001*visual_loss).mean().backward(retain_graph=True)
        local_model.train()
        tgtmodel.train()

        atklosslist = []
        cleanlosslist = []
        backdoorlosslist = []
        local_dataset_size = 0
        for batch_idx, data_labels in tqdm(enumerate(local_train_loader), total=len(local_train_loader)):
            batch = hlpr.task.get_batch(batch_idx, data_labels)
            bs = batch.batch_size
            local_dataset_size += bs

            data, target = batch.inputs, batch.labels

            # atktarget = target_transform(target) # Flipping label
            atktarget = target_transform(target, n_classes=10)

            # First, update the atkmodel weights using tgtoptimizer.step(), fix local_model weights
            # noise = tgtmodel(data) * hlpr.params.eps # Do I need to pass atktarget?
            noise = tgtmodel(data, atktarget)
            atkdata = clip_image(data + noise)

            # augmented_atkdata = post_transforms(atkdata)
            # augmented_data = post_transforms(data)

            augmented_atkdata = atkdata.clone()
            augmented_data = data.clone()

            # Calculus loss
            atkoutput = local_model(augmented_atkdata)
            atkloss = hlpr.task.criterion(atkoutput, atktarget)
            atklosslist.append(sum(atkloss))

            # local_optimizer.zero_grad()
            tgtoptimizer.zero_grad()
            # visual_loss = torch.sum(torch.square(atkdata - data), dim=(1, 2, 3))
            # (0.999*atkloss + 0.001*visual_loss).mean().backward(retain_graph=True)
            # (0.9999*atkloss + 0.0001*visual_loss).mean().backward(retain_graph=True) # exp 85

            visual_loss = 1 - torch.nn.functional.cosine_similarity(atkdata.flatten(start_dim=1), data.flatten(start_dim=1))
            (0.1*atkloss + 0.9*visual_loss).mean().backward(retain_graph=True) # quite good with 2 clients
            tgtoptimizer.step() # Only update the weights of the generative model

            

            # Second, update local_model weights using local_optimizer.step(), fix atkmodel weights
            # noise = tgtmodel(data) * hlpr.params.eps
            noise = tgtmodel(data, atktarget)
            atkdata = clip_image(data + noise)

            # augmented_atkdata = post_transforms(atkdata)
            # augmented_data = post_transforms(data)

            augmented_atkdata = atkdata.clone()
            augmented_data = data.clone()

            output = local_model(augmented_data)
            atkoutput = local_model(augmented_atkdata.detach())
            clean_loss = hlpr.task.criterion(output, target)
            cleanlosslist.append(sum(clean_loss))
            backdoor_loss = hlpr.task.criterion(atkoutput, atktarget)
            backdoorlosslist.append(sum(backdoor_loss))
            total_loss = hlpr.params.alpha * clean_loss + (1-hlpr.params.alpha) * backdoor_loss

            local_optimizer.zero_grad()
            total_loss.mean().backward()

            # Apply grad_mask before updating weights
            print(f"Apply grad mask, ratio {hlpr.params.gradmask_ratio}")
            apply_grad_mask(local_model, mask_grad_list)

            local_optimizer.step() # Only update the weights of the classifier model

            if batch_idx % 10 or batch_idx == len(local_train_loader) - 1:
                print(f"Train Malicious [Batch {batch_idx}/{len(local_train_loader)}, Epoch {local_epoch}], Generative Loss {atkloss.mean().item():.4f}, Visual Loss: {visual_loss.mean().item():.4f}, Classifier: Clean Loss {clean_loss.mean().item():.4f} " \
                    f"Backdoor Loss {backdoor_loss.mean().item():.4f} Total {total_loss.mean().item():.4f}")
        
        atkloss = sum(atklosslist) / local_dataset_size
        cleanloss = sum(cleanlosslist) / local_dataset_size
        backdoorloss = sum(backdoorlosslist) / local_dataset_size

        return atkloss, cleanloss, backdoorloss
    else:
        cleanlosslist = []
        local_dataset_size = 0
        for batch_idx, data_labels in tqdm(enumerate(local_train_loader), total=len(local_train_loader)):
            batch = hlpr.task.get_batch(batch_idx, data_labels)
            bs = batch.batch_size
            local_dataset_size += bs

            data, target = batch.inputs, batch.labels
            augmented_data = post_transforms(data)

            output = local_model(augmented_data)
            loss = hlpr.task.criterion(output, target)
            cleanlosslist.append(sum(loss))

            loss.mean().backward()
            local_optimizer.step()

            if batch_idx % 10 or batch_idx == len(local_train_loader) - 1:
                print(f"Train Benign [Batch {batch_idx}/{len(local_train_loader)}, Epoch {local_epoch}], Classifier: Clean Loss {loss.mean().item():.4f}")
        cleanloss = sum(cleanlosslist) / local_dataset_size
        return cleanloss

def train_like_a_gan_iba(hlpr: Helper, local_epoch, local_model, local_optimizer, local_train_loader, attack=True, global_model=None, 
                    atkmodel=None, tgtmodel=None, tgtoptimizer=None, target_transform=None, clip_image=None, post_transforms=None, threshold_ba=0.75):
    if attack:
        # atkmodel.eval()
        local_model.eval() # IMPORTANT
        tgtmodel.train() # IMPORTANT
        # print(id(tgtmodel))
        local_ba = 0.0
        # criterion = torch.nn.CrossEntropyLoss(reduction='none')
        while local_ba < threshold_ba:
            atklosslist = []
            local_dataset_size = 0
            backdoor_correct = 0

            for batch_idx, data_labels in tqdm(enumerate(local_train_loader), total=len(local_train_loader)):
                batch = hlpr.task.get_batch(batch_idx, data_labels)
                bs = batch.batch_size
                local_dataset_size += bs

                data, target = batch.inputs, batch.labels
                # data, target = copy.deepcopy(batch.inputs), copy.deepcopy(batch.labels)

                atktarget = target_transform(target) # Flipping label
                
                # print(atktarget)
                # atktarget = target_transform(target, n_classes=10)

                # First, update the atkmodel weights using tgtoptimizer.step(), fix local_model weights
                noise = tgtmodel(data) * hlpr.params.eps # Do I need to pass atktarget?
                # noise = tgtmodel(data, atktarget) * hlpr.params.eps
                atkdata = clip_image(data + noise)
                # import IPython; IPython.embed(); exit(0)

                # augmented_atkdata = post_transforms(atkdata)
                # augmented_data = post_transforms(data)

                augmented_atkdata = atkdata.clone()
                augmented_data = data.clone()

                # Calculus loss
                atkoutput = local_model(augmented_atkdata)
                atkloss = hlpr.task.criterion(atkoutput, atktarget)
                # atkloss = criterion(atkoutput, atktarget)
                atklosslist.append(sum(atkloss))
                # atklosslist.append(atkloss.item())

                backdoor_pred = atkoutput.max(1, keepdim=True)[1]
                backdoor_correct += backdoor_pred.eq(atktarget.view_as(backdoor_pred)).sum().item()

                # local_optimizer.zero_grad()
                tgtoptimizer.zero_grad()
                atkloss.mean().backward(retain_graph=True)
                # atkloss.backward()
                tgtoptimizer.step() # Only update the weights of the generative model

                visual_diff = torch.sum(torch.square(atkdata - data)) / bs
                if batch_idx % 10 or batch_idx == len(local_train_loader) - 1:
                    print(f"Train Generative [Batch {batch_idx}/{len(local_train_loader)}, Epoch {local_epoch}], Generative Loss {atkloss.mean().item():.4f}, Visual Difference: {visual_diff.mean().item():.4f}")

            local_ba = backdoor_correct / local_dataset_size
            print(f"Local BA: {local_ba:.4f}")
        

        local_model.train() # IMPORTANT
        tgtmodel.eval() # IMPORTANT
        cleanlosslist = []
        backdoorlosslist = []
        # local_train_loader_iter = iter(local_train_loader)
        # Second, update local_model weights using local_optimizer.step(), fix atkmodel weights
        for batch_idx, data_labels in tqdm(enumerate(local_train_loader), total=len(local_train_loader)): 
            noise = tgtmodel(data) * hlpr.params.eps
            # noise = tgtmodel(data, atktarget) * hlpr.params.eps
            atkdata = clip_image(data + noise)

            # augmented_atkdata = post_transforms(atkdata)
            # augmented_data = post_transforms(data)

            augmented_atkdata = atkdata.clone()
            augmented_data = data.clone()

            output = local_model(augmented_data)
            atkoutput = local_model(augmented_atkdata.detach())
            clean_loss = hlpr.task.criterion(output, target)
            cleanlosslist.append(sum(clean_loss))
            backdoor_loss = hlpr.task.criterion(atkoutput, atktarget)
            backdoorlosslist.append(sum(backdoor_loss))
            total_loss = hlpr.params.alpha * clean_loss + (1-hlpr.params.alpha) * backdoor_loss

            local_optimizer.zero_grad()
            total_loss.mean().backward()
            local_optimizer.step() # Only update the weights of the classifier model

            if batch_idx % 10 or batch_idx == len(local_train_loader) - 1:
                print(f"Train Malicious [Batch {batch_idx}/{len(local_train_loader)}, Epoch {local_epoch}], Classifier: Clean Loss {clean_loss.mean().item():.4f} " \
                    f"Backdoor Loss {backdoor_loss.mean().item():.4f} Total {total_loss.mean().item():.4f}")
        
        atkloss = sum(atklosslist) / local_dataset_size
        cleanloss = sum(cleanlosslist) / local_dataset_size
        backdoorloss = sum(backdoorlosslist) / local_dataset_size

        return atkloss, cleanloss, backdoorloss, local_ba
    else:
        local_model.train()
        cleanlosslist = []
        local_dataset_size = 0
        for batch_idx, data_labels in tqdm(enumerate(local_train_loader), total=len(local_train_loader)):
            batch = hlpr.task.get_batch(batch_idx, data_labels)
            bs = batch.batch_size
            local_dataset_size += bs

            data, target = batch.inputs, batch.labels
            # augmented_data = post_transforms(data)
            augmented_data = data.clone()

            output = local_model(augmented_data)
            loss = hlpr.task.criterion(output, target)
            cleanlosslist.append(sum(loss))

            local_optimizer.zero_grad()
            loss.mean().backward()
            local_optimizer.step()

            if batch_idx % 10 or batch_idx == len(local_train_loader) - 1:
                print(f"Train Benign [Batch {batch_idx}/{len(local_train_loader)}, Epoch {local_epoch}], Classifier: Clean Loss {loss.mean().item():.4f}")
        cleanloss = sum(cleanlosslist) / local_dataset_size
        return cleanloss


def train_like_a_gan(hlpr: Helper, local_epoch, local_model, local_optimizer, local_train_loader, attack=True, global_model=None, 
                    atkmodel=None, tgtmodel=None, tgtoptimizer=None, target_transform=None, clip_image=None, post_transforms=None):
    if attack:
        # atkmodel.eval()
        local_model.train()
        tgtmodel.train()

        atklosslist = []
        cleanlosslist = []
        backdoorlosslist = []
        local_dataset_size = 0
        for batch_idx, data_labels in tqdm(enumerate(local_train_loader), total=len(local_train_loader)):
            batch = hlpr.task.get_batch(batch_idx, data_labels)
            bs = batch.batch_size
            local_dataset_size += bs

            data, target = batch.inputs, batch.labels

            # atktarget = target_transform(target) # Flipping label
            atktarget = target_transform(target, n_classes=hlpr.params.num_classes)

            # First, update the atkmodel weights using tgtoptimizer.step(), fix local_model weights
            # noise = tgtmodel(data) * hlpr.params.eps # Do I need to pass atktarget?
            noise = tgtmodel(data, atktarget) * hlpr.params.eps
            atkdata = clip_image(data + noise)

            # augmented_atkdata = post_transforms(atkdata)
            # augmented_data = post_transforms(data)

            augmented_atkdata = atkdata.clone()
            augmented_data = data.clone()

            # Calculus loss
            atkoutput = local_model(augmented_atkdata)
            atkloss = hlpr.task.criterion(atkoutput, atktarget)
            atklosslist.append(sum(atkloss))

            # local_optimizer.zero_grad()
            tgtoptimizer.zero_grad()
            # atkloss.mean().backward(retain_graph=True)
            atkloss.mean().backward()
            tgtoptimizer.step() # Only update the weights of the generative model

            # visual_diff = torch.sum(torch.square(atkdata - data)) / bs
            visual_diff = 1 - torch.nn.functional.cosine_similarity(atkdata.flatten(start_dim=1), data.flatten(start_dim=1))

            # Second, update local_model weights using local_optimizer.step(), fix atkmodel weights
            # noise = tgtmodel(data) * hlpr.params.eps
            noise = tgtmodel(data, atktarget) * hlpr.params.eps
            atkdata = clip_image(data + noise)

            # augmented_atkdata = post_transforms(atkdata)
            # augmented_data = post_transforms(data)

            augmented_atkdata = atkdata.clone()
            augmented_data = data.clone()

            output = local_model(augmented_data)
            atkoutput = local_model(augmented_atkdata.detach())
            clean_loss = hlpr.task.criterion(output, target)
            cleanlosslist.append(sum(clean_loss))
            backdoor_loss = hlpr.task.criterion(atkoutput, atktarget)
            backdoorlosslist.append(sum(backdoor_loss))
            total_loss = hlpr.params.alpha * clean_loss + (1-hlpr.params.alpha) * backdoor_loss

            local_optimizer.zero_grad()
            total_loss.mean().backward()
            local_optimizer.step() # Only update the weights of the classifier model

            if batch_idx % 10 or batch_idx == len(local_train_loader) - 1:
                print(f"Train Malicious [Batch {batch_idx}/{len(local_train_loader)}, Epoch {local_epoch}], Generative Loss {atkloss.mean().item():.4f}, Visual Difference: {visual_diff.mean().item():.4f}, Classifier: Clean Loss {clean_loss.mean().item():.4f} " \
                    f"Backdoor Loss {backdoor_loss.mean().item():.4f} Total {total_loss.mean().item():.4f}")
        
        atkloss = sum(atklosslist) / local_dataset_size
        cleanloss = sum(cleanlosslist) / local_dataset_size
        backdoorloss = sum(backdoorlosslist) / local_dataset_size

        return atkloss, cleanloss, backdoorloss
    else:
        cleanlosslist = []
        local_dataset_size = 0
        for batch_idx, data_labels in tqdm(enumerate(local_train_loader), total=len(local_train_loader)):
            batch = hlpr.task.get_batch(batch_idx, data_labels)
            bs = batch.batch_size
            local_dataset_size += bs

            data, target = batch.inputs, batch.labels
            # augmented_data = post_transforms(data)
            augmented_data = data.clone()

            output = local_model(augmented_data)
            loss = hlpr.task.criterion(output, target)
            cleanlosslist.append(sum(loss))

            loss.mean().backward()
            local_optimizer.step()

            if batch_idx % 10 or batch_idx == len(local_train_loader) - 1:
                print(f"Train Benign [Batch {batch_idx}/{len(local_train_loader)}, Epoch {local_epoch}], Classifier: Clean Loss {loss.mean().item():.4f}")
        cleanloss = sum(cleanlosslist) / local_dataset_size
        return cleanloss


def train(hlpr: Helper, local_epoch, local_model, local_optimizer, local_train_loader, attack=True, global_model=None, 
          atkmodel=None, tgtmodel=None, tgtoptimizer=None, target_transform=None, clip_image=None, post_transforms=None):
    if attack:
        atkmodel.eval()
        local_model.train()
        tgtmodel.train()

        atklosslist = []
        cleanlosslist = []
        backdoorlosslist = []
        local_dataset_size = 0
        for batch_idx, data_labels in tqdm(enumerate(local_train_loader), total=len(local_train_loader)):
            batch = hlpr.task.get_batch(batch_idx, data_labels)
            bs = batch.batch_size
            local_dataset_size += bs

            data, target = batch.inputs, batch.labels
            atktarget = target_transform(target) # Flipping label

            # First, update the atkmodel weights using tgtoptimizer.step(), fix local_model weights
            noise = tgtmodel(data) * hlpr.params.eps # Do I need to pass atktarget?
            atkdata = clip_image(data + noise)

            augmented_atkdata = post_transforms(atkdata)
            augmented_data = post_transforms(data)

            # Calculus loss
            atkoutput = local_model(augmented_atkdata)
            atkloss = hlpr.task.criterion(atkoutput, atktarget)
            atklosslist.append(sum(atkloss))

            # local_optimizer.zero_grad()
            tgtoptimizer.zero_grad()
            atkloss.mean().backward(retain_graph=True)
            tgtoptimizer.step() # Only update the weights of the generative model

            # Second, update local_model weights using local_optimizer.step(), fix atkmodel weights
            noise = atkmodel(data) * hlpr.params.eps
            atkdata = clip_image(data + noise)

            augmented_atkdata = post_transforms(atkdata)
            augmented_data = post_transforms(data)

            output = local_model(augmented_data)
            atkoutput = local_model(augmented_atkdata)
            clean_loss = hlpr.task.criterion(output, target)
            cleanlosslist.append(sum(clean_loss))
            backdoor_loss = hlpr.task.criterion(atkoutput, atktarget)
            backdoorlosslist.append(sum(backdoor_loss))
            total_loss = hlpr.params.alpha * clean_loss + (1-hlpr.params.alpha) * backdoor_loss

            local_optimizer.zero_grad()
            total_loss.mean().backward()
            local_optimizer.step() # Only update the weights of the classifier model

            if batch_idx % 10 or batch_idx == len(local_train_loader) - 1:
                print(f"Train Malicious [Batch {batch_idx}/{len(local_train_loader)}, Epoch {local_epoch}], Classifier: Clean Loss {clean_loss.mean().item():.4f} " \
                    f"Backdoor Loss {backdoor_loss.mean().item():.4f} Total {total_loss.mean().item():.4f}")
        
        atkloss = sum(atklosslist) / local_dataset_size
        cleanloss = sum(cleanlosslist) / local_dataset_size
        backdoorloss = sum(backdoorlosslist) / local_dataset_size

        return atkloss, cleanloss, backdoorloss
    else:
        local_model.train()
        cleanlosslist = []
        local_dataset_size = 0
        for batch_idx, data_labels in tqdm(enumerate(local_train_loader), total=len(local_train_loader)):
            batch = hlpr.task.get_batch(batch_idx, data_labels)
            bs = batch.batch_size
            local_dataset_size += bs

            data, target = batch.inputs, batch.labels
            # augmented_data = post_transforms(data)

            augmented_data = data.clone()

            output = local_model(augmented_data)
            # output = output.logits # test with microsoft/resnet-50, remove this line for other models
            loss = hlpr.task.criterion(output, target)
            cleanlosslist.append(sum(loss))

            local_optimizer.zero_grad() # Worked well Feb 5, 2024
            # local_optimizer.zero_grad(set_to_none=True)
            loss.mean().backward()
            local_optimizer.step()

            if batch_idx % 10 or batch_idx == len(local_train_loader) - 1:
                print(f"Train Benign [Batch {batch_idx}/{len(local_train_loader)}, Epoch {local_epoch}], Classifier: Clean Loss {loss.mean().item():.4f}")
        cleanloss = sum(cleanlosslist) / local_dataset_size
        return cleanloss


def test(hlpr: Helper, epoch, backdoor=False, model=None, atkmodel=None):
    
    if model is None:
        model = hlpr.task.model # get the global model
    model.eval()

    if atkmodel:
        atkmodel.eval()
    # hlpr.task.reset_metrics()
    
    class_accuracies = {}
    class_counts= {}
    # count = 0
    for i in range(hlpr.params.num_classes):
        class_counts[i] = 0


    test_loss, correct = 0.0, 0
    test_backdoor_loss, backdoor_correct = 0.0, 0
    with torch.no_grad():
        for i, data_labels in tqdm(enumerate(hlpr.task.test_loader)):
            batch = hlpr.task.get_batch(i, data_labels)
            
            data, target = batch.inputs, batch.labels
            output = model(data)
            # print("output", output)
            # output = output.logits # test with microsoft/resnet-50, remove this line for other models

            test_loss += hlpr.task.criterion(output, target).sum().item()
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            if atkmodel:
                # target_transform = hlpr.task.target_transform
                target_transform = hlpr.task.sample_negative_labels
                # atkdata, atktarget = make_backdoor_batch(hlpr, data, target, atkmodel, target_transform, multitarget=False)
                atkdata, atktarget = make_backdoor_batch(hlpr, data, target, atkmodel, target_transform, multitarget=True)

                # visual_diff = torch.sum(torch.square(atkdata - data), dim=(1, 2, 3))
                visual_diff = 1 - torch.nn.functional.cosine_similarity(atkdata.flatten(start_dim=1), data.flatten(start_dim=1)) # cosine distance, range [0; 1], comment out to test other visual losses, this works
                # visual_diff = torch.nn.functional.mse_loss(atkdata, data, reduction="none")
                # ssim = pytorch_ssim.SSIM(window_size=11)
                # visual_diff = (ssim(atkdata, data) + 1) / 2
                # huber_loss = torch.nn.HuberLoss(reduction='none', delta=1.0)
                # visual_diff = huber_loss(atkdata, data).sum(dim=(1,2,3))

                atkoutput = model(atkdata)
                # atkoutput = atkoutput.logits # test with microsoft/resnet-50, remove this line for other models

                test_backdoor_loss += hlpr.task.criterion(atkoutput, atktarget).sum().item()
                atkpred = atkoutput.max(1, keepdim=True)[1]  # get the index of the max log-probability

                calculate_each_class_accuracy(atkpred, atktarget, class_accuracies, class_counts)
                # count += tmp

                backdoor_correct += atkpred.eq(atktarget.view_as(atkpred)).sum().item()

    test_dataset_size = len(hlpr.task.test_loader.dataset)
    test_loss /= test_dataset_size
    acc = correct / test_dataset_size

    # print("count", count)
    # n_test_examples_each_class = test_dataset_size // 10
    assert len(class_counts) in [15, 10, 200, 0, 100], "The number of classes is not equal to 15 (chestxray) or 10 (mnist, fashionmnist, cifar10) or 200 (tinyimagenet) or 100 (cifar100) or 0 (no attack). The problem is due to randomly sampling negative labels"
    # assert len(class_counts) == 10, "The number of classes is not equal to 10. The problem is due to randomly sampling negative labels"
    # print("class_accuracies", class_accuracies)
    # print("class_counts", class_counts)
    for tar, corr in class_accuracies.items():
        if class_counts[tar] != 0:
            class_accuracies[tar] = corr / class_counts[tar]

    if atkmodel:
        test_backdoor_loss /= len(hlpr.task.test_loader.dataset)
        backdoor_acc = backdoor_correct / len(hlpr.task.test_loader.dataset)

        print('\nTest [{}]: Clean Loss {:.4f}, Backdoor Loss {:.4f}, Clean Accuracy {:.4f}, Backdoor Accuracy {:.4f}, Visual Difference {:.4f}'.format(epoch,
                test_loss, test_backdoor_loss, acc, backdoor_acc, visual_diff.mean().item()))
        return acc, backdoor_acc, test_loss, test_backdoor_loss, visual_diff.mean().item(), class_accuracies
    elif not atkmodel:
        print('\nTest [{}]: Clean Loss {:.4f}'.format(epoch, test_loss))
        return acc, None, test_loss, None, None, {}
    else:
        # raise the error that informs the user atkmodel is None using raise
        raise ValueError("atkmodel is None")

    # with torch.no_grad():
    #     for i, data in tqdm(enumerate(hlpr.task.test_loader)):
    #         batch = hlpr.task.get_batch(i, data)
    #         if backdoor:
    #             batch = hlpr.attack.synthesizer.make_backdoor_batch(batch,
    #                                                                 test=True,
    #                                                                 attack=True)

    #         outputs = model(batch.inputs)
    #         hlpr.task.accumulate_metrics(outputs=outputs, labels=batch.labels)
    # metric = hlpr.task.report_metrics(epoch,
    #                          prefix=f'Backdoor {str(backdoor):5s}. Epoch: ')
    # return metric

def test_with_patch(hlpr: Helper, epoch, model=None, test=True):
    if model is None:
        model = hlpr.task.model
    model.eval()

    class_accuracies = {}
    class_counts= {}
    # count = 0
    for i in range(hlpr.params.num_classes):
        class_counts[i] = 0

    test_loss, correct = 0.0, 0
    test_backdoor_loss, backdoor_correct = 0.0, 0
    with torch.no_grad():
        for i, data_labels in tqdm(enumerate(hlpr.task.test_loader)):
            batch = hlpr.task.get_batch(i, data_labels)
            
            data, target = batch.inputs, batch.labels
            output = model(data)
            test_loss += hlpr.task.criterion(output, target).sum().item()
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            backdoored_batch = hlpr.attack.synthesizer.make_backdoor_batch(batch, test=True, attack=True, test_phase=True)
            atkdata, atktarget = backdoored_batch.inputs, backdoored_batch.labels
            atkoutput = model(atkdata)

            test_backdoor_loss += hlpr.task.criterion(atkoutput, atktarget).sum().item()
            atkpred = atkoutput.max(1, keepdim=True)[1]  # get the index of the max log-probability

            calculate_each_class_accuracy(atkpred, atktarget, class_accuracies, class_counts)

            backdoor_correct += atkpred.eq(atktarget.view_as(atkpred)).sum().item()
    test_loss /= len(hlpr.task.test_loader.dataset)
    acc = correct / len(hlpr.task.test_loader.dataset)

    assert len(class_counts) in [15, 10, 200, 0, 100], "The number of classes is not equal to 15 (chestxray) or 10 (mnist, fashionmnist, cifar10) or 200 (tinyimagenet) or 100 (cifar100) or 0 (no attack). The problem is due to randomly sampling negative labels"
    for tar, corr in class_accuracies.items():
        if class_counts[tar] != 0:
            class_accuracies[tar] = corr / class_counts[tar]

    test_backdoor_loss /= len(hlpr.task.test_loader.dataset)
    backdoor_acc = backdoor_correct / len(hlpr.task.test_loader.dataset)

    print('\nTest [{}]: Clean Loss {:.4f}, Backdoor Loss {:.4f}, Clean Accuracy {:.4f}, Backdoor Accuracy {:.4f}'.format(epoch,
            test_loss, test_backdoor_loss, acc, backdoor_acc))
    return acc, backdoor_acc, test_loss, test_backdoor_loss, class_accuracies
    # return acc, backdoor_acc, test_loss, test_backdoor_loss, None

def test_a3fl(hlpr, epoch):
    criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
    model = hlpr.task.model # get the global model
    model.eval()

    class_accuracies = {}
    class_counts= {}
    # count = 0
    for i in range(hlpr.params.num_classes):
        class_counts[i] = 0

    with torch.no_grad():
        data_source = hlpr.task.test_loader
        total_loss = 0 # for calculating clean loss
        correct = 0 # for calculating clean acc
        backdoor_correct = 0 # for calculating backdoor loss
        backdoor_loss = 0 # for calculating backdoor loss
        backdoor_correct = 0 # for calculating backdoor loss
        num_data = 0.
        for batch_id, batch in tqdm(enumerate(data_source)):
            data, targets = batch
            data, targets = data.cuda(), targets.cuda()
            output = model(data)
            total_loss += criterion(output, targets).item()
            pred = output.data.max(1)[1] 
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
                
            # backdoor part
            backdoor_data, backdoor_targets = hlpr.attack.poison_input(data, targets, eval=True)
            backdoor_output = model(backdoor_data)
            backdoor_loss += criterion(backdoor_output, backdoor_targets).item()
            backdoor_pred = backdoor_output.data.max(1)[1] 
            backdoor_correct += backdoor_pred.eq(backdoor_targets.data.view_as(pred)).cpu().sum().item()

            calculate_each_class_accuracy(backdoor_pred, backdoor_targets, class_accuracies, class_counts)

            num_data += output.size(0) 
    acc = 100.0 * (float(correct) / float(num_data)) # clean acc
    loss = total_loss / float(num_data) # clean loss

    backdoor_acc = 100.0 * (float(backdoor_correct) / float(num_data)) # backdoor acc
    backdoor_loss = backdoor_loss / float(num_data) # backdoor loss

    assert len(class_counts) in [15, 10, 200, 0, 100], "The number of classes is not equal to 15 (chestxray) or 10 (mnist, fashionmnist, cifar10) or 200 (tinyimagenet) or 100 (cifar100) or 0 (no attack). The problem is due to randomly sampling negative labels"
    for tar, corr in class_accuracies.items():
        if class_counts[tar] != 0:
            class_accuracies[tar] = corr / class_counts[tar]
    print('\nTest [{}]: Clean Loss {:.4f}, Backdoor Loss {:.4f}, Clean Accuracy {:.4f}, Backdoor Accuracy {:.4f}'.format(epoch,
            loss, backdoor_loss, acc, backdoor_acc))

    model.train()
    # return acc, backdoor_acc, loss, backdoor_loss
    return acc, backdoor_acc, loss, backdoor_loss, class_accuracies
    
def train_with_patch(hlpr: Helper, local_epoch, local_model, local_optimizer, local_train_loader, attack=True, global_model=None, post_transforms=None):
    if attack:
        local_model.train()

        # cleanlosslist = []
        backdoorlosslist = []
        local_dataset_size = 0
        for batch_idx, data_labels in tqdm(enumerate(local_train_loader), total=len(local_train_loader)):
            batch = hlpr.task.get_batch(batch_idx, data_labels)
            bs = batch.batch_size
            local_dataset_size += bs

            data, target = batch.inputs, batch.labels

            # First, update the atkmodel weights using tgtoptimizer.step(), fix local_model weights
            backdoored_batch = hlpr.attack.synthesizer.make_backdoor_batch(batch, test=True, attack=True, test_phase=False)
            atkdata = backdoored_batch.inputs
            atktarget = backdoored_batch.labels

            # augmented_atkdata = post_transforms(atkdata)
            # augmented_data = post_transforms(data)
            augmented_atkdata = atkdata.clone()

            # output = local_model(augmented_data)
            atkoutput = local_model(augmented_atkdata)
            # clean_loss = hlpr.task.criterion(output, target)
            # cleanlosslist.append(sum(clean_loss))
            backdoor_loss = hlpr.task.criterion(atkoutput, atktarget)
            backdoorlosslist.append(sum(backdoor_loss))
            # total_loss = hlpr.params.alpha * clean_loss + (1-hlpr.params.alpha) * backdoor_loss
            total_loss = backdoor_loss

            local_optimizer.zero_grad()
            total_loss.mean().backward()
            local_optimizer.step()

            if batch_idx % 10 or batch_idx == len(local_train_loader) - 1:
                # print(f"Train Malicious [Batch {batch_idx}/{len(local_train_loader)}, Epoch {local_epoch}], Classifier: Clean Loss {clean_loss.mean().item():.4f} " \
                #     f"Backdoor Loss {backdoor_loss.mean().item():.4f} Total {total_loss.mean().item():.4f}")
                print(f"Train Malicious [Batch {batch_idx}/{len(local_train_loader)}, Epoch {local_epoch}], Classifier: Backdoor Loss {backdoor_loss.mean().item():.4f}")
        
        # cleanloss = sum(cleanlosslist) / local_dataset_size
        backdoorloss = sum(backdoorlosslist) / local_dataset_size

        # return cleanloss, backdoorloss
        return backdoorloss
    else:
        cleanlosslist = []
        local_dataset_size = 0
        for batch_idx, data_labels in tqdm(enumerate(local_train_loader), total=len(local_train_loader)):
            batch = hlpr.task.get_batch(batch_idx, data_labels)
            bs = batch.batch_size
            local_dataset_size += bs

            data, target = batch.inputs, batch.labels
            augmented_data = post_transforms(data)

            output = local_model(augmented_data)
            loss = hlpr.task.criterion(output, target)
            cleanlosslist.append(sum(loss))

            loss.mean().backward()
            local_optimizer.step()

            if batch_idx % 10 or batch_idx == len(local_train_loader) - 1:
                print(f"Train Benign [Batch {batch_idx}/{len(local_train_loader)}, Epoch {local_epoch}], Classifier: Clean Loss {loss.mean().item():.4f}")
        cleanloss = sum(cleanlosslist) / local_dataset_size
        return cleanloss

def train_malicious_a3fl(hlpr, participant_id, model, epoch):
    attacker_criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
    lr = get_lr_a3fl(hlpr, epoch)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
        momentum=0.9,
        weight_decay=0.0005)
    clean_model = copy.deepcopy(model)
    for internal_epoch in range(hlpr.params.attacker_retrain_times):
        total_loss = 0.0
        for inputs, labels in hlpr.task.fl_train_loaders[participant_id]:
            inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = hlpr.attack.poison_input(inputs, labels)
            output = model(inputs)
            loss = attacker_criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def train_benign_a3fl(hlpr, participant_id, model, epoch):
    criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
    lr = get_lr_a3fl(hlpr, epoch)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
        momentum=0.9,
        weight_decay=0.0005)
    for internal_epoch in range(hlpr.params.retrain_times):
        total_loss = 0.0
        for inputs, labels in hlpr.task.fl_train_loaders[participant_id]:
            inputs, labels = inputs.cuda(), labels.cuda()
            output = model(inputs)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def run_fl_round(hlpr: Helper, epoch, atkmodels_dict, history_grad_list_neurotoxin):
    global_model = hlpr.task.model
    global_model.train()
    local_model = hlpr.task.local_model
    round_participants = hlpr.task.sample_users_for_round(epoch)
    weight_accumulator = hlpr.task.get_empty_accumulator()

    participated_clients = {} # Map user id to its local model
    malicious_local_models = {}

    if hlpr.params.attack_type == "a3fl":
        first_adversary = 0
        hlpr.task.copy_params(global_model, local_model)
        hlpr.attack.search_trigger(local_model, hlpr.task.fl_train_loaders[first_adversary], 'outter', first_adversary, epoch)

    for user in tqdm(round_participants):
        hlpr.task.copy_params(global_model, local_model)
        local_optimizer = hlpr.task.make_optimizer(local_model)
        post_transforms = PostTensorTransform(hlpr.params).to(hlpr.params.device)
        
        participated_clients[user.user_id] = local_model

        if user.compromised:
            malicious_local_models[user.user_id] = local_model
            atkmodel, tgtmodel, tgtoptimizer = atkmodels_dict[user.user_id]
            # atkmodel.train() # Starts from exp 66
            # target_transform = hlpr.task.target_transform
            assert hlpr.params.attack_type in ["venomancer", "patch", "a3fl"]
            if hlpr.params.attack_type == "venomancer":
                target_transform = hlpr.task.sample_negative_labels
            # elif hlpr.params.attack_type == "a3fl":
                # target_transform = hlpr.task.target_transform # Option for 1 target label
                # target_transform = hlpr.task.sample_negative_labels
            
            # mask_grad_list = get_grad_mask(hlpr, local_model, local_optimizer, user.train_loader, history_grad_list_neurotoxin, ratio=hlpr.params.gradmask_ratio)

            logger.warning(f"Compromised user: {user.user_id} in run_fl_round function {epoch}")
            for local_epoch in tqdm(range(hlpr.params.fl_poison_epochs)):
                # atkloss, cleanloss, backdoorloss = train(hlpr, local_epoch, local_model, local_optimizer, user.train_loader, attack=True, global_model=global_model,
                #                                          atkmodel=atkmodel, tgtmodel=tgtmodel, tgtoptimizer=tgtoptimizer, target_transform=target_transform,
                #                                          clip_image=hlpr.task.clip_image, post_transforms=post_transforms)
                
                # atkloss, cleanloss, backdoorloss = train_like_a_gan(hlpr, local_epoch, local_model, local_optimizer, user.train_loader, attack=True, global_model=global_model,
                #                                          atkmodel=atkmodel, tgtmodel=tgtmodel, tgtoptimizer=tgtoptimizer, target_transform=target_transform,
                #                                          clip_image=hlpr.task.clip_image, post_transforms=post_transforms)
                
                # atkloss, cleanloss, backdoorloss, local_ba = train_like_a_gan_iba(hlpr, local_epoch, local_model, local_optimizer, user.train_loader, attack=True, global_model=global_model,
                #                                          atkmodel=atkmodel, tgtmodel=tgtmodel, tgtoptimizer=tgtoptimizer, target_transform=target_transform,
                #                                          clip_image=hlpr.task.clip_image, post_transforms=post_transforms, threshold_ba=0.85)
                
                
                # atkloss, cleanloss, backdoorloss = train_like_a_gan_with_visual_loss_check_durability(hlpr, local_epoch, local_model, local_optimizer, user.train_loader, attack=True, global_model=global_model,
                #                                          atkmodel=atkmodel, tgtmodel=tgtmodel, tgtoptimizer=tgtoptimizer, target_transform=target_transform,
                #                                          clip_image=hlpr.task.clip_image, post_transforms=post_transforms, mask_grad_list=mask_grad_list)
                
                # atkloss, cleanloss, backdoorloss = train_like_a_gan_learnable_eps(hlpr, local_epoch, local_model, local_optimizer, user.train_loader, attack=True, global_model=global_model,
                #                                          atkmodel=atkmodel, tgtmodel=tgtmodel, tgtoptimizer=tgtoptimizer, target_transform=target_transform,
                #                                          clip_image=hlpr.task.clip_image, post_transforms=post_transforms, mask_grad_list=mask_grad_list)
                
                # atkloss, cleanloss, backdoorloss = train_like_a_gan_clean_label(hlpr, local_epoch, local_model, local_optimizer, user.train_loader, attack=True, global_model=global_model,
                #                                          atkmodel=atkmodel, tgtmodel=tgtmodel, tgtoptimizer=tgtoptimizer, target_transform=target_transform,
                #                                          clip_image=hlpr.task.clip_image, post_transforms=post_transforms)
                
                # atkloss, cleanloss, backdoorloss = train_with_noise_patch(hlpr, local_epoch, local_model, local_optimizer, user.train_loader, attack=True, global_model=global_model,
                #                                             atkmodel=atkmodel, tgtmodel=tgtmodel, tgtoptimizer=tgtoptimizer, target_transform=target_transform,
                #                                             clip_image=hlpr.task.clip_image, post_transforms=post_transforms)
                if hlpr.params.attack_type == "venomancer":
                    atkloss, cleanloss, backdoorloss = train_like_a_gan_with_visual_loss(hlpr, local_epoch, local_model, local_optimizer, user.train_loader, attack=True, global_model=global_model,
                                                            atkmodel=atkmodel, tgtmodel=tgtmodel, tgtoptimizer=tgtoptimizer, target_transform=target_transform,
                                                            clip_image=hlpr.task.clip_image, post_transforms=post_transforms)
                    atkmodel.load_state_dict(tgtmodel.state_dict())
                elif hlpr.params.attack_type == "a3fl":
                    train_malicious_a3fl(hlpr, user.user_id, local_model, epoch)
                elif hlpr.params.attack_type == "patch":
                    backdoorloss = train_with_patch(hlpr, local_epoch, local_model, local_optimizer, user.train_loader, attack=True, global_model=global_model)

                # cleanloss, backdoorloss = train_with_patch(hlpr, local_epoch, local_model, local_optimizer, user.train_loader, attack=True, global_model=global_model, post_transforms=post_transforms)
                # backdoorloss = train_with_patch(hlpr, local_epoch, local_model, local_optimizer, user.train_loader, attack=True, global_model=global_model, post_transforms=post_transforms)

            local_update = hlpr.attack.get_fl_update(local_model, global_model)
            for name in local_update.keys():
                local_update[name] *= hlpr.params.fl_weight_scale
        else:
            if user.user_id in atkmodels_dict.keys():
                malicious_local_models[user.user_id] = local_model

            logger.warning(f"Non-compromised user: {user.user_id} in run_fl_round function {epoch}")
            for local_epoch in range(hlpr.params.fl_local_epochs):
                if hlpr.params.attack_type == "venomancer":
                    cleanloss = train(hlpr, local_epoch, local_model, local_optimizer, user.train_loader, attack=False, global_model=global_model, post_transforms=post_transforms)
                elif hlpr.params.attack_type == "patch":
                    cleanloss = train_with_patch(hlpr, local_epoch, local_model, local_optimizer, user.train_loader, attack=False, global_model=global_model, post_transforms=post_transforms)
                elif hlpr.params.attack_type == "a3fl":
                    train_benign_a3fl(hlpr, user.user_id, local_model, epoch)
        
            local_update = hlpr.attack.get_fl_update(local_model, global_model)
        # hlpr.save_update(model=local_update, userID=user.user_id)

        # Don't save local updates to file, save to memory instead
        hlpr.task.adding_local_updated_model(local_update=local_update, user_id=user.user_id)

        # if user.compromised:
        #     hlpr.attack.local_dataset = deepcopy(user.train_loader)
    # atkmodel_avg, tgtmodel_avg, tgtoptimizer_avg = aggregate_atkmodels(hlpr, atkmodels_dict, round_participants)
    if atkmodels_dict:
        print("Picking the best atkmodel")
        best_atkmodel, best_tgtmodel, best_tgtoptimizer, local_backdoor_acc = pick_best_atkmodel(hlpr, atkmodels_dict, round_participants, malicious_local_models)

    # Apply defenses here (before aggregation)
    if hlpr.params.defense.lower() == "norm_clipping":
        print("Apply norm clipping") # DEBUG
        hlpr.defense.clip_weight_diff()
        hlpr.defense.aggr(weight_accumulator, global_model)
        hlpr.task.update_global_model(weight_accumulator, global_model)
    elif hlpr.params.defense.lower() == "krum":
        print(f"Apply Krum, mode {hlpr.params.mode_krum}") # DEBUG
        hlpr.defense.run(participated_clients)
        hlpr.defense.aggr(weight_accumulator, global_model)
        hlpr.task.update_global_model(weight_accumulator, global_model)
    elif hlpr.params.defense.lower() == "rlr":
        print(f"Apply RLR") # DEBUG
        hlpr.defense.run(global_model, participated_clients)
        hlpr.defense.aggr(weight_accumulator, global_model)
        hlpr.task.update_global_model(weight_accumulator, global_model)
    elif hlpr.params.defense.lower() == "deepsight":
        print(f"Apply Deepsight") # DEBUG
        hlpr.defense.aggr(weight_accumulator, global_model)
        hlpr.task.update_global_model(weight_accumulator, global_model)
    elif hlpr.params.defense.lower() == "fedrad":
        print(f"Apply FedRAD. Required distillation knowledge on server") # DEBUG
        hlpr.defense.aggr(weight_accumulator, global_model) # After this, global_model gets updated
    elif hlpr.params.defense.lower() == "rflbat":
        print(f"Apply RFLBAT") # DEBUG
        hlpr.defense.aggr(weight_accumulator, global_model)
        hlpr.task.update_global_model(weight_accumulator, global_model)
    elif hlpr.params.defense.lower() == "fedavg":
        print(f"Apply FedAvg") # DEBUG
        hlpr.defense.aggr(weight_accumulator, global_model)
        hlpr.task.update_global_model(weight_accumulator, global_model)
    # hlpr.attack.perform_attack(global_model, epoch)
    # hlpr.defense.aggr(weight_accumulator, global_model) # use this if don't want to repeat like above

    # import IPython; IPython.embed(); exit();
    # hlpr.task.update_global_model(weight_accumulator, global_model) # use this if don't want to repeat like above

    # Some defenses can be applied here (after aggregation)
    if hlpr.params.defense.lower() == "weak_dp":
        print("Apply weak DP") # DEBUG
        # hlpr.defense.clip_weight_diff()
        hlpr.defense.add_noise_to_weights(global_model)

    # return atkmodel_avg, tgtmodel_avg, tgtoptimizer_avg
    if atkmodels_dict:
        return best_atkmodel, best_tgtmodel, best_tgtoptimizer, local_backdoor_acc
    else:
        return None, None, None, None

def run(hlpr: Helper):
    decay = False
    atkmodels_dict = {} # Store atkmodel, tgtmodel, tgtoptimizer in a dictionary
    for user_id in hlpr.task.adversaries:
        atkmodel_init, tgtmodel_init, tgtoptimizer_init = hlpr.task.get_atkmodel()
        atkmodels_dict[user_id] = [atkmodel_init, tgtmodel_init, tgtoptimizer_init]
    
    history_grad_list_neurotoxin = [] # Store grad_list for neurotoxin attack

    class_accuracies_log = {} # logging accuracy for each class on wandb
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        logger.info(f"Communication round {epoch}")
        atkmodel, tgtmodel, tgtoptimizer, local_backdoor_acc = run_fl_round(hlpr, epoch, atkmodels_dict, history_grad_list_neurotoxin)

        # atkmodel.eval() # Starts from exp66
        if hlpr.params.attack_type == "venomancer":
            clean_acc, backdoor_acc, clean_loss, backdoor_loss, visual_diff, class_accuracies = test(hlpr, epoch, backdoor=True, atkmodel=tgtmodel) # Use tgtmodel (currently in the eval mode)
        elif hlpr.params.attack_type == "patch":
            clean_acc, backdoor_acc, clean_loss, backdoor_loss, class_accuracies = test_with_patch(hlpr, epoch, test=True)
        elif hlpr.params.attack_type == "a3fl":
            clean_acc, backdoor_acc, clean_loss, backdoor_loss, class_accuracies = test_a3fl(hlpr, epoch) # multi-targets
            # clean_acc, backdoor_acc, clean_loss, backdoor_loss = test_a3fl(hlpr, epoch) # 1 target label
            
        else:
            raise ValueError("Invalid attack type")
        # wandb log acc and backdoor_acc, clean_loss, backdoor_loss
        if hlpr.params.attack_type == "venomancer":
            wandb.log({"Clean Accuracy": clean_acc, "Backdoor Accuracy": backdoor_acc, "Clean Loss": clean_loss, "Backdoor Loss": backdoor_loss, "Visual Difference": visual_diff}, step=epoch)
        elif hlpr.params.attack_type in ["patch", "a3fl"]:
            wandb.log({"Clean Accuracy": clean_acc, "Backdoor Accuracy": backdoor_acc, "Clean Loss": clean_loss, "Backdoor Loss": backdoor_loss}, step=epoch)

        for target, accuracy in class_accuracies.items():
            class_accuracies_log[f"Class {target}"] = accuracy
        wandb.log(class_accuracies_log, step=epoch)

        # hlpr.record_accuracy(metric, test(hlpr, epoch, backdoor=True), epoch)

        # hlpr.save_model(hlpr.task.model, epoch, metric)
        # if backdoor_acc >= 0.8 and not decay:
        #     decay = True
        #     start_decay_epoch = epoch
        
        # if decay:
        #     hlpr.params.eps = max(0.05, hlpr.params.eps*(1 - 0.001)**(epoch - start_decay_epoch))

        hlpr.save_model(hlpr.task.model, tgtmodel, epoch)

if __name__ == '__main__':
    # print('aslkdhjaskdjaskdj')
    parser = argparse.ArgumentParser(description='Backdoors in Federated Learning')
    parser.add_argument('--params', dest='params', help="Parameters of the task. Pass in the path to config file", required=True)
    parser.add_argument('--name', dest='name', help="Name of a task (MNIST, CIFAR-10, etc)",required=True)
    parser.add_argument('--time', dest='time', help='Current time from the terminal', required=True)
    parser.add_argument('--exp', dest='exp', help='Experiment number', required=True)
    args = parser.parse_args()
    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    # params['current_time'] = datetime.now().strftime('%m.%d_%H.%M.%S')
    params['current_time'] = args.time # Get the current time from the terminal
    params['name'] = args.name # Name of the running task
    params['exp'] = args.exp # Experiment number

    helper = Helper(params)
     # Transform original labels to target labels, default: all2one
    file_path = ".wandb_key"
    with open(file_path, "r") as f:
        key = f.readline().strip()

    # You need to initialize your wandb HERE
    wandb.login(key=key)
    wandb.init(project="backdoor-attack", entity="nguyenhongsonk62hust", name=f"{params['exp']}_{params['name']}-{params['current_time']}", dir="./hdd/home/ssd_data/Son/Venomancer/wandb/wandb")
    logger.warning(create_table(params)) # Print the table of parameters to the terminal, showing as warnings
    try:
        run(helper)
    except KeyboardInterrupt:
        if helper.params.log:
            answer = prompt('\nDelete the repo? (y/n): ')
            if answer in ['Y', 'y', 'yes']:
                logger.error(f"Fine. Deleted: {helper.params.folder_path}")
                prefix = helper.params.prefix
                if os.path.exists(helper.params.folder_path):
                    shutil.rmtree(helper.params.folder_path)

                if os.path.exists(prefix + f"/saved_latest_atkmodel/{helper.params.current_time}"):
                    shutil.rmtree(prefix + f"/saved_latest_atkmodel/{helper.params.current_time}")

                # Check if a file path exists
                if os.path.exists(prefix + f"/dataloaders/poison_loader_{helper.params.current_time}.pkl"):
                    os.remove(prefix + f"/dataloaders/poison_loader_{helper.params.current_time}.pkl")

                if os.path.exists(prefix + f"/dataloaders/train_loaders_{helper.params.current_time}.pkl"):
                    os.remove(prefix + f"/dataloaders/train_loaders_{helper.params.current_time}.pkl")
            else:
                logger.error(f"Aborted training. "
                             f"Results: {helper.params.folder_path}, \
                             ./saved_latest_atkmodel/{helper.params.current_time}, \
                             ./dataloaders/poison_loader_{helper.params.current_time}.pkl, \
                             ./dataloaders/train_loaders_{helper.params.current_time}.pkl. ")
        else:
            logger.error(f"Aborted training. No output generated.")
