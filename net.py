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
    def __init__(self, out_dim, p=0):
        super().__init__()

        input_size  = 384
        hidden1     = 128
        hidden2     = 128
        hidden3     = 64

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),
            nn.BatchNorm2d(8),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=2, padding=0),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),
            nn.BatchNorm2d(16),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=2, padding=0),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[0, 0]),
            nn.BatchNorm2d(16),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=2, padding=0),
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

            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.Mish(),
            nn.Dropout(p),

            nn.Linear(hidden3, out_dim),
        )