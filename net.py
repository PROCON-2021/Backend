import numpy as np
import torch.nn as nn
import torch as t

class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()

    def calc_out_size(self, in_size, kernel_size, stride, padding):
        return int(np.floor(((in_size + (2*padding) - (kernel_size-1) - 1) // stride) + 1))

    def forward(self, x):

        x = self.conv(x)  # conv.
        x = t.flatten(x, 1)  # flatten
        x = self.dense(x)  # dense

        return x

class Conv2dModel(ConvModel):
    def __init__(self, fft_size, out_dim, p=0, mode='abs'):
        super().__init__()

        if mode == 'abs':
            if fft_size==64: input_size = 2880
            elif fft_size==128: input_size = 2880
        elif mode == 'shoulders' or mode == 'thighs':
            if fft_size==64: input_size = 3168
            elif fft_size==128: input_size = 2880

        hidden1 = 128
        hidden2 = 128

        out_channels = [16, 32, 32, 32]

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=out_channels[0], kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
            nn.BatchNorm2d(out_channels[0]),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p),

            nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
            nn.BatchNorm2d(out_channels[1]),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p),

            nn.Conv2d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),
            nn.BatchNorm2d(out_channels[2]),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=3, padding=0),
            nn.Dropout2d(p),

            nn.Conv2d(in_channels=out_channels[2], out_channels=out_channels[3], kernel_size=[2, 2], stride=[1, 1], padding=[0, 0]),
            nn.BatchNorm2d(out_channels[3]),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=3, padding=0),
            nn.Dropout2d(p),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=out_channels[0], kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
            nn.BatchNorm2d(out_channels[0]),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p),

            nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
            nn.BatchNorm2d(out_channels[1]),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p),

            nn.Conv2d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),
            nn.BatchNorm2d(out_channels[2]),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=3, padding=0),
            nn.Dropout2d(p),

            nn.Conv2d(in_channels=out_channels[2], out_channels=out_channels[3], kernel_size=[2, 2], stride=[1, 1], padding=[0, 0]),
            nn.BatchNorm2d(out_channels[3]),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=3, padding=0),
            nn.Dropout2d(p),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=out_channels[0], kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
            nn.BatchNorm2d(out_channels[0]),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p),

            nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),
            nn.BatchNorm2d(out_channels[1]),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p),

            nn.Conv2d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),
            nn.BatchNorm2d(out_channels[2]),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=3, padding=0),
            nn.Dropout2d(p),

            nn.Conv2d(in_channels=out_channels[2], out_channels=out_channels[3], kernel_size=[2, 2], stride=[1, 1], padding=[0, 0]),
            nn.BatchNorm2d(out_channels[3]),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=3, padding=0),
            nn.Dropout2d(p),
        )

        self.dense = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.Mish(),
            nn.Dropout(p),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.Mish(),
            nn.Dropout(p),

            nn.Linear(hidden2, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.Mish(),
            nn.Dropout(p),

            nn.Linear(hidden2, out_dim),
        )

    def forward(self, x):
        
        x1 = self.conv1(x[:,0:1,:,:])
        x2 = self.conv2(x[:,1:2,:,:])
        x3 = self.conv3(x[:,2:3,:,:])

        x1 = t.flatten(x1, 1)  
        x2 = t.flatten(x2, 1) 
        x3 = t.flatten(x3, 1) 

        l = t.cat([x1, x2, x3], dim=1)

        y = self.dense(l)  # dense

        return y