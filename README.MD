# Venomancer: Towards Imperceptible and Target-on-Demand Backdoor Attack in Federated Learning

## Venomancer framework
![image](images/readme/venomancer_diagram.svg)

## Installation
- Python 3.10.10, PyTorch 2.0.0, conda 23.11.0, wandb 0.16.1

- All experiments are done on a single machine with 252GB RAM, 64 Intel Xeon Gold 6242 CPUs @ 2.80GHz, and 6 NVIDIA RTX A5000 GPUs with 24GB RAM each. The utilized OS is Ubuntu 20.04.6 LTS

- Set up a `conda` environment by running `conda create --name <envname> --file requirements.txt` and activate it by running `conda activate <envname>`

- Read `training.py` file and add to the place that requires the `wandb` API key.

## How to run the experiments?
- MNIST
    - `export CUDA_VISIBLE_DEVICES=<gpu_id> && mytime=$(date "+%m.%d_%H.%M.%S") && mkdir -p $(pwd)"/experiments/exp<exp_id>/"$mytime && python training.py --name mnist --params configs/mnist_fed.yaml --time $mytime --exp exp<exp_id>> $(pwd)"/experiments/exp<exp_id>/"$mytime"/logs.txt"`
- CIFAR-10
    - `export CUDA_VISIBLE_DEVICES=<gpu_id> && mytime=$(date "+%m.%d_%H.%M.%S") && mkdir -p $(pwd)"/experiments/exp<exp_id>/"$mytime && python training.py --name cifar10 --params configs/cifar10_fed.yaml --time $mytime --exp exp<exp_id> > $(pwd)"/experiments/exp<exp_id>/"$mytime"/logs.txt"`
