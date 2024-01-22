import torch
import torch.nn as nn

from copy import deepcopy

class MNISTAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 64, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 3, stride=2),  # b, 16, 5, 5
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.BatchNorm2d(1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class MNISTConditionalAutoencoder(nn.Module):
    def __init__(self, n_classes=10, input_dim=28):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.linear_c = nn.Linear(n_classes, 1 * input_dim * input_dim)
        self.n_classes = n_classes
        self.input_dim = input_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1+1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 64, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 3, stride=2),  # b, 16, 5, 5
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.BatchNorm2d(1),
            nn.Tanh()
        )

    def forward(self, x, c):
        # c = self.linear_c(self.label_emb(c)).view(-1, 1, self.input_dim, self.input_dim)
        # x = torch.cat([x, c], dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x    
    
class ConditionalAutoencoder(nn.Module):
    def __init__(self, n_classes, input_dim):
        super().__init__()
        
        self.label_emb = nn.Embedding(n_classes, n_classes)
        # self.label_emb = nn.Embedding(n_classes, n_classes*10)
        self.linear_c = nn.Linear(n_classes, 1 * input_dim * input_dim)
        # self.linear_c = nn.Linear(n_classes*10, 1 * input_dim * input_dim)
        self.n_classes = n_classes
        self.input_dim = input_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3+1, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x, c):
        c = self.linear_c(self.label_emb(c)).view(-1, 1, self.input_dim, self.input_dim)
        x = torch.cat([x, c], dim=1)
        
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
# Toggle this if you want to use the conditional autoencoder WITH a fixed pattern embedded
# class ConditionalAutoencoder(nn.Module):
#     def __init__(self, n_classes, input_dim, pattern_tensor):
#         super().__init__()

#         self.pattern_tensor = pattern_tensor.flatten().view(1, -1)
        
#         self.linear = nn.Linear(self.pattern_tensor.shape[1], 1 * input_dim * input_dim)
#         self.n_classes = n_classes
#         self.input_dim = input_dim

#         self.encoder = nn.Sequential(
#             nn.Conv2d(3 + 1, 16, 4, stride=2, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(True),
#             nn.Conv2d(16, 32, 4, stride=2, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(True),
#             nn.Conv2d(32, 64, 4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             nn.Conv2d(64, 128, 4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True)
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         pattern = self.linear(self.pattern_tensor.cuda()).view(-1, 1, self.input_dim, self.input_dim).repeat(x.shape[0], 1, 1, 1).cuda()
#         x = torch.cat([x, pattern], dim=1)
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
    

