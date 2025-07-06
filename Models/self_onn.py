from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import math

import torch.nn.functional as F
from fastonn.SelfONN import SelfONN1d
from torchinfo import summary
from torch.autograd import Variable



class MyDropout2(nn.Module):
    def __init__(self, p=0.5):
        super(MyDropout2, self).__init__()
        self.p = p
        if self.p < 1:
            self.multiplier_ = 1.0 / (1.0-p)
        else:
            self.multiplier_ = 0.0
    def forward(self, input):
        if not self.training:
            return input
        selected_ = torch.Tensor(input.shape).uniform_(0,1)>self.p
        mea = torch.mean(input)
        sele_ = input > mea
        selected_ = sele_
        aa = torch.sum(sele_)
        # bb = aa.numpy()/ input.__len__()
       # print(bb)
        #selected_num = torch.sum(selected_)
        #if selected_num > 0:
        #    multiplier_ = torch.numel(selected_)/selected_num
        #else:
        #    multiplier_ = 0
        if input.is_cuda:
            selected_ = Variable(selected_.type(torch.cuda.FloatTensor), requires_grad=False)
        else:
            selected_ = Variable(selected_.type(torch.FloatTensor), requires_grad=False)
        return torch.mul(selected_,input) * self.multiplier_




# class self_onn(nn.Module):
#     def __init__(self):
#         super(self_onn, self).__init__()
        
#         self.onn1 = nn.Sequential(
#             SelfONN1d(in_channels=2, out_channels=32, kernel_size=9,  padding=4,q=3),
#             # nn.BatchNorm1d(32),
#             nn.Tanh(),
#             nn.MaxPool1d(kernel_size=2, stride=2),

#             SelfONN1d(in_channels=32, out_channels=32, kernel_size=7,  padding=3,q=3),
#             # nn.BatchNorm1d(32),
#             nn.Tanh(),
#             nn.MaxPool1d(kernel_size=2, stride=2),
#             # MultiSpectralAttentionLayer(channel=32, dct_length=900, reduction=16, n = 16),
#         )
#         self.onn2 = nn.Sequential(
#             SelfONN1d(in_channels=32, out_channels=32, kernel_size=5, padding=2,q=3),
#             # nn.BatchNorm1d(32),
#             nn.Tanh(),
#             nn.MaxPool1d(kernel_size=2, stride=2),

#             SelfONN1d(in_channels=32, out_channels=32, kernel_size=3, padding=1,q=3),
#             # nn.BatchNorm1d(32),
#             nn.Tanh(),
#             nn.MaxPool1d(kernel_size=2, stride=2),
#             # MultiSpectralAttentionLayer(channel=32, dct_length=225, reduction=16, n = 16),
#         )
#         self.onn3 = nn.Sequential(

#             SelfONN1d(in_channels=32, out_channels=32, kernel_size=3, padding=1,q=4),
#             # nn.BatchNorm1d(32),
#             nn.Tanh(),
#             nn.MaxPool1d(kernel_size=2, stride=2),


#             SelfONN1d(in_channels=32, out_channels=32, kernel_size=3, padding=1,q=4),
#             # nn.BatchNorm1d(32),
#             nn.Tanh(),
#             nn.MaxPool1d(kernel_size=2, stride=2),

#         )

#         self.onn4 = nn.Sequential(

#             SelfONN1d(in_channels=32, out_channels=32, kernel_size=3, padding=1,q=4),
#             # nn.BatchNorm1d(32),
#             nn.Tanh(),
#             nn.MaxPool1d(kernel_size=2, stride=2),

#             # SelfONN1d(in_channels=32, out_channels=32, kernel_size=3, padding=1,q=4),
#             # # nn.BatchNorm1d(32),
#             # nn.Tanh(),
#             # nn.MaxPool1d(kernel_size=2, stride=2),
#             # MultiSpectralAttentionLayer(channel=32, dct_length=14, reduction=16, n = 8),
#         )



#         self.classifier = nn.Sequential(
#             nn.Linear(896, 64),
#             nn.GELU(),
#             nn.ReLU(),
#             MyDropout2(p = 0.5),
#             nn.Linear(64, 1),
#             nn.Sigmoid()
#         )


#     def forward(self, x):   # torch.Size([10, 2, 10000])

        
#         x = self.onn1(x)
#         # print(x.shape)
#         x = self.onn2(x)
#         # print(x.shape)
#         x = self.onn3(x)
#         # print(x.shape)
#         x = self.onn4(x)

#         x = torch.reshape(x, (10, -1))
#         # print(x.shape)
#         x = self.classifier(x)
#         return x



class self_onn(nn.Module):
    def __init__(self):
        super(self_onn, self).__init__()
        self.onn = nn.Sequential(

            SelfONN1d(in_channels=2, out_channels=32, kernel_size=9, padding=4,q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            SelfONN1d(in_channels=32, out_channels=32, kernel_size=7, padding=3,q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            SelfONN1d(in_channels=32, out_channels=32, kernel_size=5, padding=2,q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            SelfONN1d(in_channels=32, out_channels=32, kernel_size=3, padding=1,q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            SelfONN1d(in_channels=32, out_channels=32, kernel_size=3, padding=1,q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            SelfONN1d(in_channels=32, out_channels=32, kernel_size=3, padding=1,q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            SelfONN1d(in_channels=32, out_channels=64, kernel_size=3, padding=1,q=4),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            SelfONN1d(in_channels=64, out_channels=64, kernel_size=3, padding=1,q=4),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            SelfONN1d(in_channels=64, out_channels=64, kernel_size=3, padding=1,q=4),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            
        )
        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(1216, 64),
            nn.GELU(),
            nn.ReLU(),
            # MyDropout2(0.5),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):   # torch.Size([10, 2, 10000])
        x = self.onn(x)
        # x = torch.reshape(x, (10, -1))
        x = self.flatten(x)
        # print(x.shape)
        x = self.classifier(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        # x = self.sigmoid(x)

        return x
        


if __name__ == '__main__':
    model = self_onn()
    summary(model, (1, 2, 10000))