import torch
import torch.nn as nn
import torch.nn.functional as F

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self, out_channel):
        super().__init__()

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=True)
 
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
 
        self.conv_last = nn.Sequential(
            nn.Conv2d(64, out_channel, 1),
            nn.BatchNorm2d(out_channel),
        )
 
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
 
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
 
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
 
        x = self.dconv_down4(x)
 
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
 
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
 
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
 
        x = self.dconv_up1(x)
 
        out = self.conv_last(x)
 
        out = torch.tanh(out)
 
        return out


class ConditionalUNet(nn.Module):

    def __init__(self, n_classes, input_dim, out_channel):
        super().__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.linear_c = nn.Linear(n_classes, 1 * input_dim * input_dim)
        self.n_classes = n_classes
        self.input_dim = input_dim

        self.dconv_down1 = double_conv(3 + 1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Sequential(
            nn.Conv2d(64, out_channel, 1),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x, c):
        c = self.linear_c(self.label_emb(c)).view(-1, 1, self.input_dim, self.input_dim)
        x = torch.cat([x, c], dim=1)
        
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        out = F.tanh(out)

        return out
    

class ChestXRayConditionalUNet(nn.Module): # ChestXRay

    def __init__(self, n_classes, input_dim, out_channel):
        super().__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.linear_c = nn.Linear(n_classes, 1 * input_dim * input_dim)
        self.n_classes = n_classes
        self.input_dim = input_dim

        self.dconv_down1 = double_conv(1 + 1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Sequential(
            nn.Conv2d(64, out_channel, 1),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x, c):
        c = self.linear_c(self.label_emb(c)).view(-1, 1, self.input_dim, self.input_dim)
        x = torch.cat([x, c], dim=1)
        
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        out = F.tanh(out)

        return out
    

class FashionMNISTConditionalUNet(nn.Module): # FashionMNIST

    def __init__(self, n_classes, input_dim, out_channel):
        super().__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.linear_c = nn.Linear(n_classes, 1 * input_dim * input_dim)
        self.n_classes = n_classes
        self.input_dim = input_dim

        # self.conv0 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv0 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=3, bias=False) # (2, 28, 28) -> (2, 32, 32)

        self.dconv_down1 = double_conv(1 + 1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Sequential(
            nn.Conv2d(64, out_channel, kernel_size=5, padding=0, stride=1),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x, c):
        c = self.linear_c(self.label_emb(c)).view(-1, 1, self.input_dim, self.input_dim)
        x = torch.cat([x, c], dim=1)
        
        x = self.conv0(x) # (2, 32, 32)

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        out = F.tanh(out)

        return out
    

class MNISTConditionalUNet(nn.Module): # FashionMNIST

    def __init__(self, n_classes, input_dim, out_channel):
        super().__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.linear_c = nn.Linear(n_classes, 1 * input_dim * input_dim)
        self.n_classes = n_classes
        self.input_dim = input_dim

        # self.conv0 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv0 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=3, bias=False) # (2, 28, 28) -> (2, 32, 32)

        self.dconv_down1 = double_conv(1 + 1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Sequential(
            nn.Conv2d(64, out_channel, kernel_size=5, padding=0, stride=1),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x, c):
        c = self.linear_c(self.label_emb(c)).view(-1, 1, self.input_dim, self.input_dim)
        x = torch.cat([x, c], dim=1)
        
        x = self.conv0(x) # (2, 32, 32)

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        out = F.tanh(out)

        return out
