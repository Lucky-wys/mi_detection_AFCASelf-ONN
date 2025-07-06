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
        if input.is_cuda:
            selected_ = Variable(selected_.type(torch.cuda.FloatTensor), requires_grad=False)
        else:
            selected_ = Variable(selected_.type(torch.FloatTensor), requires_grad=False)
        return torch.mul(selected_,input) * self.multiplier_


class FilterSelector(nn.Module):
    def __init__(self, channel, num_freq, len_seq, n):
        super(FilterSelector, self).__init__()
        self.num_freq = num_freq
        self.len_seq = len_seq
        self.n = n
        self.channel = channel
        self.weights = nn.Parameter(torch.ones(num_freq))
        
    def forward(self, filters):

        device = filters.device
        # 动态创建权重矩阵
        weighted = filters * self.weights.view(-1, 1).to(device)  # 保持维度对齐
        
        # 选择前n个滤波器
        _, top_indices = torch.topk(self.weights, self.n)
        selected = weighted[top_indices]
        
        # 创建目标矩阵并确保设备一致
        weight_filter = torch.zeros(self.channel, self.len_seq, 
                                   dtype=selected.dtype, 
                                   device=device)
        
        # 均匀分配通道
        chunk_size = self.channel // self.n
        for i in range(self.n):
            start = i * chunk_size
            end = (i+1) * chunk_size
            if i == self.n-1:  # 处理余数
                end = self.channel
            weight_filter[start:end] = selected[i]
            
        return weight_filter

def get_freq_indices(length):
    indices = list(range(length))
    return indices[-length:]  # 返回最后 num 个索引
class MultiSpectralDCTLayer(nn.Module):
    def __init__(self, length, mapper, channel, n=8):
        super(MultiSpectralDCTLayer, self).__init__()
        self.num_freq = len(mapper)
        self.register_buffer('base_weight', self.get_dct_filter(length, mapper, channel))  # 固定基础滤波器
        self.filter_selector = FilterSelector(channel=channel, num_freq=self.num_freq, len_seq=length, n=n)

    def forward(self, x):
        # 始终使用基础滤波器进行选择
        selected_weight = self.filter_selector(self.base_weight)  # 每次重新选择
        x = x * selected_weight.unsqueeze(0)  # 添加batch维度
        result = torch.sum(x, dim=2)
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size, mapper, channel):
        dct_filter = torch.zeros(tile_size, tile_size)
        for i, u in enumerate(mapper):
            for t in range(tile_size):
                dct_filter[i, t] = self.build_filter(t, u, tile_size)   # 滤波器的总大小为 length          
        return dct_filter


class MultiSpectralAttentionLayer(nn.Module):
    def __init__(self, channel, dct_length, reduction=16, n = 8):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_length = dct_length

        mapper_x = get_freq_indices(dct_length)
        self.num_split = len(mapper_x)

        self.dct_layer = MultiSpectralDCTLayer(dct_length, mapper_x, channel,n= n)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        n, c, l = x.shape
        # 适应性池化为目标长度
        if l != self.dct_length:
            x_pooled = torch.nn.functional.adaptive_avg_pool1d(x, self.dct_length)
        else:
            x_pooled = x
        
        y = self.dct_layer(x_pooled)
        y = self.fc(y).view(n, c, 1)

        
        return x * y.expand_as(x)


class self_onn_fcanet_ada_10000(nn.Module):
    def __init__(self):
        super(self_onn_fcanet_ada_10000, self).__init__()
        self.onn1 = nn.Sequential(

            SelfONN1d(in_channels=2, out_channels=32, kernel_size=9, padding=4,q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            SelfONN1d(in_channels=32, out_channels=32, kernel_size=9, padding=4,q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),

        )
        self.onn2 = nn.Sequential(
            SelfONN1d(in_channels=32, out_channels=32, kernel_size=7, padding=3,q=3),
            
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            SelfONN1d(in_channels=32, out_channels=32, kernel_size=5, padding=2,q=3),
           
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            MultiSpectralAttentionLayer(channel=32, dct_length=625,reduction=16, n = 16), 
        )

        self.onn3 = nn.Sequential(
            SelfONN1d(in_channels=32, out_channels=32, kernel_size=3, padding=1,q=4),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            SelfONN1d(in_channels=32, out_channels=32, kernel_size=3, padding=1,q=4),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            MultiSpectralAttentionLayer(channel=32, dct_length=156,reduction=16, n = 8), 
        )
        self.onn4 = nn.Sequential(
            SelfONN1d(in_channels=32, out_channels=32, kernel_size=3, padding=1,q=4),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            SelfONN1d(in_channels=32, out_channels=32, kernel_size=3, padding=1,q=4),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            MultiSpectralAttentionLayer(channel=32, dct_length=39,reduction=16, n = 4),
        )

        self.classifier = nn.Sequential(
            nn.Linear(1248, 64),
            nn.GELU(),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):   # torch.Size([10, 2, 10000])
        x1 = self.onn1(x)
        x2 = self.onn2(x1)
        x3 = self.onn3(x2)
        x4 = self.onn4(x3)

        x = torch.reshape(x4, (10, -1))
        x = self.classifier(x)
        return x
        

    

if __name__ == '__main__':
    model = self_onn_fcanet_ada_10000()
    summary(model, (10, 2, 10000))