import wandb
import os
import re

from tqdm import tqdm

# Experiments on fl_dirichlet_phi (data heteroginity)
# run_id_pool = {"exp383": "nguyenhongsonk62hust/backdoor-attack/rdqxnzt1", 
#                "exp382": "nguyenhongsonk62hust/backdoor-attack/56zbrouh",
#                "exp390": "nguyenhongsonk62hust/backdoor-attack/go9i8vs7",
#                "exp380": "nguyenhongsonk62hust/backdoor-attack/myzdc47v",
#                "exp381": "nguyenhongsonk62hust/backdoor-attack/8v8vhm9e",
#                "exp389": "nguyenhongsonk62hust/backdoor-attack/dysrsp59",
#                "exp384": "nguyenhongsonk62hust/backdoor-attack/dnf7mg32",
#                "exp385": "nguyenhongsonk62hust/backdoor-attack/ch77jh5u",
#                "exp391": "nguyenhongsonk62hust/backdoor-attack/mvhmubnv",
#                "exp386": "nguyenhongsonk62hust/backdoor-attack/fyew0fm2",
#                "exp387": "nguyenhongsonk62hust/backdoor-attack/795djn0u",
#                "exp418": "nguyenhongsonk62hust/backdoor-attack/sn13zggu",
#                "exp257": "nguyenhongsonk62hust/backdoor-attack/l6mcnc88",
#                }

# Experiments on different classifiers (VGG11, ResNet-18)
# run_id_pool = {"exp376": "nguyenhongsonk62hust/backdoor-attack/xe5b23dy", 
#                "exp379": "nguyenhongsonk62hust/backdoor-attack/9xrslg0j",
#                "exp377": "nguyenhongsonk62hust/backdoor-attack/yblrwlp6",
#                "exp388": "nguyenhongsonk62hust/backdoor-attack/ykm3c458",
#                }

# Experiments on different generators (U-Net, Autoencoder)
# run_id_pool = {"exp364": "nguyenhongsonk62hust/backdoor-attack/q23xlkx5", 
#                "exp417": "nguyenhongsonk62hust/backdoor-attack/ukd5ryyh",
#                "exp365": "nguyenhongsonk62hust/backdoor-attack/lmmmps2h",
#                "exp367": "nguyenhongsonk62hust/backdoor-attack/bcd3tk7o",
#                }

# Experiments on different fixed_frequency
# run_id_pool = {"exp393": "nguyenhongsonk62hust/backdoor-attack/fimppfen", 
#                "exp396": "nguyenhongsonk62hust/backdoor-attack/g46h4fhr",
#                "exp399": "nguyenhongsonk62hust/backdoor-attack/0980lkse",
#                "exp402": "nguyenhongsonk62hust/backdoor-attack/0j95qc2m",
#                "exp394": "nguyenhongsonk62hust/backdoor-attack/f53qqfg4",
#                "exp397": "nguyenhongsonk62hust/backdoor-attack/1tq20crd",
#                "exp400": "nguyenhongsonk62hust/backdoor-attack/7npkltiu",
#                "exp403": "nguyenhongsonk62hust/backdoor-attack/x3lelyxk",
#                "exp395": "nguyenhongsonk62hust/backdoor-attack/55reg2lc",
#                "exp398": "nguyenhongsonk62hust/backdoor-attack/tmgshqmj",
#                "exp401": "nguyenhongsonk62hust/backdoor-attack/hz7aw0m5",
#                "exp404": "nguyenhongsonk62hust/backdoor-attack/w8l5iwkq",
#                }

# Experiments on the number of malicious clients P
# run_id_pool = {"exp405": "nguyenhongsonk62hust/backdoor-attack/i4x29cf5", 
#                "exp408": "nguyenhongsonk62hust/backdoor-attack/uv2iz18i",
#                "exp411": "nguyenhongsonk62hust/backdoor-attack/009seomm",
#                "exp414": "nguyenhongsonk62hust/backdoor-attack/fhaesbsw",
#                "exp406": "nguyenhongsonk62hust/backdoor-attack/agynzfd2",
#                "exp409": "nguyenhongsonk62hust/backdoor-attack/ci3vmbj0",
#                "exp412": "nguyenhongsonk62hust/backdoor-attack/0pwnzzkx",
#                "exp415": "nguyenhongsonk62hust/backdoor-attack/m5yz9mu2",
#                "exp407": "nguyenhongsonk62hust/backdoor-attack/vx747s8y",
#                "exp410": "nguyenhongsonk62hust/backdoor-attack/v4civm5y",
#                "exp413": "nguyenhongsonk62hust/backdoor-attack/64wkx3pi",
#                "exp416": "nguyenhongsonk62hust/backdoor-attack/m2594l92",
#                }

# run_id_pool = {"exp243": "nguyenhongsonk62hust/backdoor-attack/w92c49jw", 
#                }

# Defenses
# run_id_pool = {
#                "exp231": "nguyenhongsonk62hust/backdoor-attack/s8jwbet3",
#                "exp232": "nguyenhongsonk62hust/backdoor-attack/rbbskqgq",
#                "exp233": "nguyenhongsonk62hust/backdoor-attack/1l5o8ujy",
#                "exp234": "nguyenhongsonk62hust/backdoor-attack/nt4n9m3a",
#                "exp235": "nguyenhongsonk62hust/backdoor-attack/b4xwqqrd",
#                "exp236": "nguyenhongsonk62hust/backdoor-attack/njri4f35",
#                "exp237": "nguyenhongsonk62hust/backdoor-attack/pccd60p1",
#                "exp238": "nguyenhongsonk62hust/backdoor-attack/g2kxn1fh",
#                "exp277": "nguyenhongsonk62hust/backdoor-attack/qioxk7zj",
#                "exp278": "nguyenhongsonk62hust/backdoor-attack/1hsb1gna",
#                "exp279": "nguyenhongsonk62hust/backdoor-attack/qes14w2y",
#                "exp280": "nguyenhongsonk62hust/backdoor-attack/2b57gde9",
#                }

# Epsilon
# run_id_pool = {
#                "exp334": "nguyenhongsonk62hust/backdoor-attack/d37wch3r",
#                "exp336": "nguyenhongsonk62hust/backdoor-attack/yw2vkl82",
#                "exp337": "nguyenhongsonk62hust/backdoor-attack/u23pqn5k",
#                }

# run_id_pool = {"exp338": "nguyenhongsonk62hust/backdoor-attack/2nifmkvu",}

# Baseline
# run_id_pool = {"exp358": "nguyenhongsonk62hust/backdoor-attack/n1fodtmd",
#                "exp368": "nguyenhongsonk62hust/backdoor-attack/kni10hu9",
#                "exp369": "nguyenhongsonk62hust/backdoor-attack/u3ontx86",
#                "exp370": "nguyenhongsonk62hust/backdoor-attack/8fstky7n",
#                "exp371": "nguyenhongsonk62hust/backdoor-attack/x27wswko",
#                }

# Varying beta, durability
run_id_pool = {"exp270": "nguyenhongsonk62hust/backdoor-attack/a0gyinqm",
               "exp274": "nguyenhongsonk62hust/backdoor-attack/pj7q8i93",
               "exp281": "nguyenhongsonk62hust/backdoor-attack/a4modc69",
               "exp282": "nguyenhongsonk62hust/backdoor-attack/9n29hyav",
               "exp284": "nguyenhongsonk62hust/backdoor-attack/plobeov8",
               "exp285": "nguyenhongsonk62hust/backdoor-attack/hmj5ln5u",
               }

# Initialize the API
api = wandb.Api()

# Add a file to the artifact
filepath = "/home/ubuntu/son.nh/Venomancer/hdd/home/ssd_data/Son/Venomancer/saved_models"

folder_names = os.listdir(filepath)

for run_id in tqdm(run_id_pool.keys()):
    run = api.run(run_id_pool[run_id])
    time_to_find = run.name.split('-')[-1]

    wandb.login(key="917b44927c77ee61ea91005724c9bd9b470f116a")
    wandb.init(id=run_id_pool[run_id].split('/')[-1], resume='allow', project="backdoor-attack", entity="nguyenhongsonk62hust", dir="./hdd/home/ssd_data/Son/Venomancer/wandb/wandb")
    
    # Create an artifact for the model
    artifact = wandb.Artifact(run_id, type='model')

    pattern = rf".*{re.escape(time_to_find)}.*"

    # Search for the pattern in the list of folder names
    matching_folders = [folder for folder in folder_names if re.search(pattern, folder)]

    folder_name = matching_folders[0]
    
    model_pool = ["log.txt", "params.yaml.txt", "model_epoch_700.pt.tar", "model_epoch_900.pt.tar"]
    for model in model_pool:
        artifact.add_file(os.path.join(filepath, folder_name, model)) # Replace with your model file path
        print('Adding file...')

    # Log the artifact to the run
    wandb.log_artifact(artifact)