from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import math

import torch.nn.functional as F
from fastonn.SelfONN import SelfONN1d
from torchinfo import summary
from torch.autograd import Variable


class SEBlock(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)


class Self_onn_SE2(nn.Module):
    def __init__(self):
        super(Self_onn_SE2, self).__init__()

        self.onn1 = nn.Sequential(
            SelfONN1d(in_channels=2, out_channels=32, kernel_size=15, padding=7, q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            SelfONN1d(in_channels=32, out_channels=32, kernel_size=11, padding=5, q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # FECAM(2500),
            # SEBlock(32),
        )
        self.SE1 = SEBlock(32)
        self.onn2 = nn.Sequential(
            SelfONN1d(in_channels=32, out_channels=32, kernel_size=7, padding=3, q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            SelfONN1d(in_channels=32, out_channels=32, kernel_size=5, padding=2, q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # FECAM(625),
            # SEBlock(32),
        )
        self.SE2 = SEBlock(32)

        self.onn3 = nn.Sequential(
            SelfONN1d(in_channels=32, out_channels=32, kernel_size=3, padding=1, q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            #  FECAM(312),
            SelfONN1d(in_channels=32, out_channels=32, kernel_size=3, padding=1, q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # FECAM(156),
            # MultiSpectralAttentionLayer(32, dct_length=64, freq_sel_method='low16')
            # SEBlock(32),
        )
        self.SE3 = SEBlock(32)

        self.onn4 = nn.Sequential(
            SelfONN1d(in_channels=32, out_channels=32, kernel_size=3, padding=1, q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # MultiSpectralAttentionLayer(channel=32, dct_length=78, reduction=16, n = 8),
            # FECAM(78),
            SelfONN1d(in_channels=32, out_channels=32, kernel_size=3, padding=1, q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            #   FECAM(39),
            # SEBlock(32),
        )
        self.SE4 = SEBlock(32)

        self.classifier = nn.Sequential(
            nn.Linear(608, 64), nn.GELU(), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.flatten = nn.Flatten()

    def forward(self, x):  # torch.Size([10, 2, 10000])
        x1 = self.onn1(x)
        x1 = self.SE1(x1)
        # print(x1.shape)

        x2 = self.onn2(x1)
        # print(x2.shape)
        x2 = self.SE2(x2)

        x3 = self.onn3(x2)
        # print(x3.shape)
        x3 = self.SE3(x3)

        x4 = self.onn4(x3)
        # print(x4.shape)
        x4 = self.SE4(x4)

        x = self.flatten(x4)
        # print(x.shape)
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    model = Self_onn_SE2()
    summary(model, (1, 2, 5000))
