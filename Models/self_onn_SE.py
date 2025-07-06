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
        self.avg_pool = nn.AdaptiveAvgPool1d(1) 
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)  
        y = self.fc(y).view(b, c, 1)  
        return x * y.expand_as(x)
class self_onn_SE(nn.Module):
    def __init__(self):
        super(self_onn_SE, self).__init__()
        
        self.onn1 = nn.Sequential(
            SelfONN1d(in_channels=2, out_channels=32, kernel_size=9,  padding=4,q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            SelfONN1d(in_channels=32, out_channels=32, kernel_size=7,  padding=3,q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            SEBlock(32),
        )
        self.onn2 = nn.Sequential(
            SelfONN1d(in_channels=32, out_channels=32, kernel_size=5, padding=2,q=3),

            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            SelfONN1d(in_channels=32, out_channels=32, kernel_size=3, padding=1,q=3),

            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            SEBlock(32),
        )
        self.onn3 = nn.Sequential(

            SelfONN1d(in_channels=32, out_channels=32, kernel_size=3, padding=1,q=4),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            SelfONN1d(in_channels=32, out_channels=32, kernel_size=3, padding=1,q=4),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            SEBlock(32),
        )

        self.onn4 = nn.Sequential(

            SelfONN1d(in_channels=32, out_channels=64, kernel_size=3, padding=1,q=4),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            SelfONN1d(in_channels=64, out_channels=64, kernel_size=3, padding=1,q=4),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
             SEBlock(64),
        )

        self.onn5 = nn.Sequential(
            SelfONN1d(in_channels=64, out_channels=64, kernel_size=3, padding=1,q=4),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            SEBlock(64),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(1216, 64),
            nn.GELU(),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )


    def forward(self, x):   # torch.Size([10, 2, 10000])

        
        x = self.onn1(x)
        x = self.onn2(x)
        x = self.onn3(x)
        x = self.onn4(x)

        x = self.onn5(x)

        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    
if __name__ == '__main__':
    model = self_onn_SE()
    summary(model, input_size=(1, 2, 10000))