import torch
import numpy as np
import os
import cv2
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import os, glob
import random


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dx = torch.tensor([x2.size(3) - x1.size(3)])
        dy = torch.tensor([x2.size(2) - x1.size(2)])

        x1 = F.pad(x1, [dx // 2, dx - dx // 2,
                        dy // 2, dy - dy // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # x: [batch_size*channels*w*h] 2*1*512*512
        self.input = DoubleConv(n_channels, 64) # 2*64*508*508
        self.downsample1 = DownSample(64, 128)
        self.downsample2 = DownSample(128, 256)
        self.downsample3 = DownSample(256, 512)
        self.downsample4 = DownSample(512, 1024)
        self.upsample1 = UpSample(1024, 512)
        self.upsample2 = UpSample(512, 256)
        self.upsample3 = UpSample(256, 128)
        self.upsample4 = UpSample(128, 64)
        self.output = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.downsample1(x1)
        x3 = self.downsample2(x2)
        x4 = self.downsample3(x3)
        x5 = self.downsample4(x4)
        x = self.upsample1(x5, x4)
        x = self.upsample2(x, x3)
        x = self.upsample3(x, x2)
        x = self.upsample4(x, x1)
        pred = self.output(x)
        return pred

class Smear(Dataset):
    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.imagesPath = glob.glob(os.path.join(dataPath, 'images/*.BMP'))
        
    def augment(self, image, flipMode):
        flipImg = cv2.flip(image, flipMode)
        return flipImg
    
    def __len__(self):
        return len(self.imagesPath)

    def __getitem__(self, idx):
        imagePath = self.imagesPath[idx]
        labelPath = imagePath.replace('images', 'labels')
        labelPath = labelPath.replace('.BMP', '-d.bmp')
        image = cv2.imread(imagePath)
        label = cv2.imread(labelPath)

        image = cv2.resize(image, (512, 512))
        label = cv2.resize(label, (512, 512))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        
        if label.max() > 1:
            label = label / 255
        flipMode = random.choice([-1, 0, 1, 2])
        if flipMode != 2:
            image = self.augment(image, flipMode)
            label = self.augment(label, flipMode)
        return image, label
    
def train_model(model, dataPath, epochs=40, batch_size=2, lr=1e-5):
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    loss_fn = nn.BCEWithLogitsLoss()
    bestLoss = float('inf')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        
        for i, (image, label) in enumerate(train_loader):
            image, label = image.float().to(device), label.to(device)
            
            pred = model(image)
            loss = loss_fn(pred, label)
            print("epoch: ", epoch, "iteration: ", i, "loss: ", loss.item())
            if loss < bestLoss:
                bestLoss = loss
                torch.save(model.state_dict(), "weights/segmentation")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == '__main__':

    train_path = "smear-segmentation/train"
    train_data = Smear(train_path)

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                            batch_size=2,
                                            shuffle=True)

    img, lab = iter(train_loader).next()
    img.shape, lab.shape

    model = UNet(n_channels=1, n_classes=1)
    train_model(model, dataPath=train_path)