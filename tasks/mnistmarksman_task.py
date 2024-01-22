import random
from torch.utils.data import Subset
import torch.utils.data as torch_data
import torchvision
from torchvision.transforms import transforms

# from models.simple import SimpleNet, NetC_MNIST
from models.lenet import LeNet
from tasks.task import Task



class MNISTMarksmanTask(Task):
    # normalize = transforms.Normalize((0.1307,), (0.3081,))
    normalize = transforms.Normalize((0.5,), (0.5,)) # Marksman
    resize = transforms.Resize((32, 32)) # Marksman

    def load_data(self):
        self.load_mnist_data()
        number_of_samples = []
        if self.params.fl_sample_dirichlet:
            # sample indices for participants using Dirichlet distribution
            # split = min(self.params.fl_total_participants / 20, 1)
            split = 1.0
            all_range = list(range(int(len(self.train_dataset) * split)))
            
            if self.params.only_target_examples:
                print("DEBUG: Only target examples for poison_dataset")
                # Get 1000 examples having the target label
                indices = []
                for i in range(len(self.train_dataset)):
                    if self.train_dataset[i][1] == self.params.target_label[0]:
                        indices.append(i)
                        if len(indices) == 1000:
                            break
                poison_dataset = Subset(self.train_dataset, indices)
                # Create a dataloader for poisonset
                poison_loader = torch_data.DataLoader(poison_dataset,
                                                  batch_size=self.params.batch_size,
                                                  shuffle=True,
                                                  num_workers=0)
            else:
                print("DEBUG: Randomly sample 1000 examples for poison_dataset")
                # Get randomly 1000 examples over the trainset
                indices = random.sample(all_range, 1000)
                poison_dataset = Subset(self.train_dataset, indices)
                poison_loader = torch_data.DataLoader(poison_dataset,
                                                    batch_size=self.params.batch_size,
                                                    shuffle=True,
                                                    num_workers=0)
                
                
            # Save the poison_loader to dataloaders folder
            import pickle
            # Get the full path of the dataloaders folder
            import os
            dataloaders_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/dataloaders'
            path = os.path.join(dataloaders_folder, f'poison_loader_{self.params.current_time}.pkl')

            with open(path, 'wb') as f:
                print("Saving poison_loader to disk...")
                pickle.dump(poison_loader, f)
            
            self.train_dataset = Subset(self.train_dataset, all_range)
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params.fl_total_participants,
                alpha=self.params.fl_dirichlet_alpha)
            # print("DEBUG: ", [len(indices) for indices in indices_per_participant.values()])
            train_loaders, number_of_samples = zip(*[self.get_train(indices) for pos, indices in
                             indices_per_participant.items()])
        else:
            # sample indices for participants that are equally
            # split = min(self.params.fl_total_participants / 20, 1)
            split = 1.0
            all_range = list(range(int(len(self.train_dataset) * split)))
            # self.train_dataset = Subset(self.train_dataset, all_range)
            random.shuffle(all_range)
            train_loaders, number_of_samples = zip(*[self.get_train_old(all_range, pos)
                                            for pos in
                                            range(self.params.fl_total_participants)])
            # train_loaders = [self.get_iid_train_loader(all_range, pos)
            #                  for pos in
            #                  range(self.params.fl_total_participants)]
        self.fl_train_loaders = train_loaders
        self.fl_number_of_samples = number_of_samples
        
        # Save train_loaders to disk for later use (e.g. seeing the data distribution)
        import pickle
        # Get the full path of the dataloaders folder
        import os
        dataloaders_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/dataloaders'
        path = os.path.join(dataloaders_folder, f'train_loaders_{self.params.current_time}.pkl')
        with open(path, 'wb') as f:
            print("Saving train_loaders to disk...")
            pickle.dump(train_loaders, f)
        return

    def load_mnist_data(self):
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])
        # transform_train = transforms.Compose([
        #     transforms.Grayscale(num_output_channels=3),
        #     transforms.ToTensor(),
        #     self.normalize, self.resize
        # ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])
        # transform_test = transforms.Compose([
        #     transforms.Grayscale(num_output_channels=3),
        #     transforms.ToTensor(),
        #     self.normalize, self.resize
        # ])

        self.train_dataset = torchvision.datasets.MNIST(
            root=self.params.data_path,
            train=True,
            download=True,
            transform=transform_train)
        self.train_loader = torch_data.DataLoader(self.train_dataset,
                                                  batch_size=self.params.batch_size,
                                                  shuffle=True,
                                                  num_workers=0)
        self.test_dataset = torchvision.datasets.MNIST(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform_test)
        self.test_loader = torch_data.DataLoader(self.test_dataset,
                                                 batch_size=self.params.test_batch_size,
                                                 shuffle=False,
                                                 num_workers=0)
        self.classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        return True

    def build_model(self):
        # return NetC_MNIST(channels=3)
        # return SimpleNet(channels=3, num_classes=10)
        return LeNet()
    