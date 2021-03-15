import torch
import numpy as np
import os
import cv2
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
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

if __name__ == '__main__':
    # model = UNet(n_channels=1, n_classes=1)
    # model.load_state_dict(torch.load("weights/seg-model.pth", map_location=torch.device(device)))
    # model.eval()

    # data_dir = 'separated-data'
    # classes = ["im_Dyskeratotic", "im_Koilocytotic", "im_Metaplastic", "im_Parabasal", "im_Superficial-Intermediate"]

    # for cell in classes:
    #     cell_path = os.path.join(data_dir, cell)
    #     files = os.listdir(cell_path)
    #     files = [os.path.join(cell_path, f) for f in files if f.endswith('.bmp')]

    #     for f in files:
    #         img = cv2.imread(f)
    #         img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #         img = img.reshape(1, 1, img.shape[0], img.shape[1])
    #         img_tensor = torch.from_numpy(img)
    #         img_tesnor = img_tensor.to(device=device, dtype=torch.float32)

    #         pred = model(img_tesnor)
    #         pred = np.array(pred.data.cpu()[0][0])
    #         pred[pred >= 0.5] = 255
    #         pred[pred < 0.5] = 0

    #         cv2.imwrite(os.path.join(cell_path, f), pred)

    # img = cv2.imread('separated-data/test/im_Dyskeratotic/005.bmp')
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img = img.reshape(1, 1, img.shape[0], img.shape[1])
    # img_tensor = torch.from_numpy(img)
    # img_tesnor = img_tensor.to(device=device, dtype=torch.float32)

    # pred = model(img_tesnor)
    # pred = np.array(pred.data.cpu()[0][0])
    # pred[pred >= 0.5] = 255
    # pred[pred < 0.5] = 0

    # cv2.imwrite('results.png', pred)

    import cv2
    import numpy as np

    # opencv loads the image in BGR, convert it to RGB
    img = cv2.cvtColor(cv2.imread('separated-data/test/im_Dyskeratotic/001.bmp'),
                    cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold input image as mask
    
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    cv2.imwrite('results.png', thresh)