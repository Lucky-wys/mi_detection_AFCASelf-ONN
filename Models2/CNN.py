import torch
import torch.nn as nn
import numpy as np
import math

import torch.nn.functional as F
from fastonn.SelfONN import SelfONN1d
from torchinfo import summary
from torch.autograd import Variable


from torch.autograd import Variable


class MyDropout2(nn.Module):
    def __init__(self, p=0.5):
        super(MyDropout2, self).__init__()
        self.p = p
        if self.p < 1:
            self.multiplier_ = 1.0 / (1.0 - p)
        else:
            self.multiplier_ = 0.0

    def forward(self, input):
        if not self.training:
            return input
        selected_ = torch.Tensor(input.shape).uniform_(0, 1) > self.p
        mea = torch.mean(input)
        sele_ = input > mea
        selected_ = sele_
        aa = torch.sum(sele_)

        if input.is_cuda:
            selected_ = Variable(selected_.type(torch.cuda.FloatTensor), requires_grad=False)
        else:
            selected_ = Variable(selected_.type(torch.FloatTensor), requires_grad=False)
        return torch.mul(selected_, input) * self.multiplier_


from torchinfo import summary


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=15, stride=2, padding=7)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=2, padding=5)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=2, padding=3)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.conv5 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv8 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.linear1 = nn.Linear(1280, 64)
        self.linear2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(32)
        self.bn5 = nn.BatchNorm1d(32)
        self.bn6 = nn.BatchNorm1d(32)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.3)
        self.faltten = nn.Flatten()

    def forward(self, x):
        x = self.bn1(F.relu((self.conv1(x))))
        x = self.bn2(F.relu((self.conv2(x))))
        x = self.bn3(F.relu((self.conv3(x))))
        x = self.bn4(F.relu((self.conv4(x))))
        x = self.bn5(F.relu((self.conv5(x))))
        x = self.bn6(F.relu((self.conv6(x))))
        x = self.bn7(F.relu((self.conv7(x))))
        x = self.bn8(F.relu((self.conv8(x))))
        x = self.faltten(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.sigmoid(x)

        return x


if __name__ == "__main__":
    model = CNN()
    summary(model, input_size=(1, 2, 5000))
