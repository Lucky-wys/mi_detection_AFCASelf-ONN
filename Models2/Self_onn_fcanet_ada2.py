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


class FilterSelector(nn.Module):
    """
    自适应频率选择器 - 基于可训练权重动态选择最重要的频率分量
    """

    def __init__(self, channel, num_freq, len_seq, n, tau=1.0):
        super(FilterSelector, self).__init__()
        self.num_freq = num_freq
        self.len_seq = len_seq
        self.n = n
        self.channel = channel
        self.tau = tau  # 温度参数，控制 softmax 的尖锐度

        # 使用 Xavier 初始化 + 小的随机扰动，确保权重不同
        self.weights = nn.Parameter(torch.empty(num_freq))
        nn.init.xavier_uniform_(self.weights.unsqueeze(0))
        self.weights.data = self.weights.data.squeeze(0) + torch.randn(num_freq) * 0.01

    def forward(self, filters):
        """
        Args:
            filters: [num_freq, len_seq] - DCT 基础滤波器

        Returns:
            weight_filter: [channel, len_seq] - 加权后的滤波器
        """
        device = filters.device
        batch_size = filters.shape[0] if filters.dim() > 2 else 1

        #  使用 softmax 归一化权重，形成概率分布 (确保权重不同)
        # softmax(x) 使权重在 (0, 1) 之间，且和为 1
        normalized_weights = torch.softmax(self.weights / self.tau, dim=0)  # [num_freq]

        # 选择权重最大的 n 个频率分量
        _, top_indices = torch.topk(normalized_weights, self.n)

        #  提取选中的滤波器及其权重
        selected_filters = filters[top_indices]  # [n, len_seq]
        selected_weights = normalized_weights[top_indices]  # [n]

        #  将选中的 n 个频率分量分配给 channel 个通道
        # 方式: 循环分配，确保每个通道得到不同频率的组合
        weight_filter = torch.zeros(self.channel, self.len_seq, dtype=selected_filters.dtype, device=device)

        chunk_size = self.channel // self.n
        for i in range(self.n):
            # 使用加权后的滤波器 (权重作为缩放因子)
            weighted_filter = selected_filters[i] * selected_weights[i]  # [len_seq]
            start = i * chunk_size
            end = (i + 1) * chunk_size
            if i == self.n - 1:  # 处理余数
                end = self.channel
            weight_filter[start:end] = weighted_filter.unsqueeze(0).expand(end - start, -1)

        return weight_filter


def get_freq_indices(length):
    indices = list(range(length))
    return indices[-length:]


class MultiSpectralDCTLayer(nn.Module):
    """
    多光谱 DCT 层 - 自适应提取频率特征
    """

    def __init__(self, length, mapper, channel, n=8, num_heads=1):
        super(MultiSpectralDCTLayer, self).__init__()
        self.num_freq = len(mapper)
        self.num_heads = num_heads
        self.length = length

        # 生成固定的 DCT 基础滤波器 (不训练，但用于初始化)
        base_weight = self.get_dct_filter(length, mapper, channel)
        self.register_buffer("base_weight", base_weight)

        # 创建多个 FilterSelector (多头机制，可选)
        self.filter_selectors = nn.ModuleList(
            [
                FilterSelector(channel=channel, num_freq=self.num_freq, len_seq=length, n=n, tau=1.0 / (i + 1))
                for i in range(num_heads)
            ]
        )

        # 可学习的头权重融合
        if num_heads > 1:
            self.head_weights = nn.Parameter(torch.ones(num_heads) / num_heads)

    def forward(self, x):
        """
        Args:
            x: [batch, channel, length]

        Returns:
            result: [batch, channel] - 每个通道的特征向量
        """
        # 自适应选择频率分量
        if self.num_heads == 1:
            selected_weight = self.filter_selectors[0](self.base_weight)
        else:
            # 多头融合
            selected_weights = []
            for selector in self.filter_selectors:
                w = selector(self.base_weight)
                selected_weights.append(w)

            # 加权融合多头输出
            selected_weight = torch.stack(selected_weights, dim=0)  # [num_heads, channel, length]
            head_weights = torch.softmax(self.head_weights, dim=0)
            selected_weight = (selected_weight * head_weights.view(-1, 1, 1)).sum(dim=0)

        # 应用加权滤波器并聚合
        x = x * selected_weight.unsqueeze(0)  # [batch, channel, length]
        result = torch.sum(x, dim=2)  # [batch, channel] ✓ 每个通道得到不同的权重

        return result

    def build_filter(self, pos, freq, POS):
        """构建 DCT 基础滤波器"""
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size, mapper, channel):
        """生成固定的 DCT 滤波器矩阵"""
        length = len(mapper)
        dct_filter = torch.zeros(length, tile_size)
        for i, u in enumerate(mapper):
            for t in range(tile_size):
                dct_filter[i, t] = self.build_filter(t, u, tile_size)
        return dct_filter


class MultiSpectralAttentionLayer(nn.Module):
    """
    多光谱自适应注意力层 - 基于自适应频率选择的通道注意力
    1. 通过自适应 DCT 层提取频率域特征
    2. 使用 FC 网络为每个通道生成独立权重
    3. 支持两个改进版本：
       - v1: 标准版 - 直接 FC 处理
       - v2: 增强版 - 添加 Batch Norm 和 Dropout 稳定训练
    """

    def __init__(self, channel, dct_length, base_length, reduction=16, n=16, version="v2"):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_length = dct_length
        self.version = version

        mapper_x = get_freq_indices(base_length)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_length // base_length) for temp_x in mapper_x]

        # 使用改进的 MultiSpectralDCTLayer (支持自适应频率选择)
        self.dct_layer = MultiSpectralDCTLayer(dct_length, mapper_x, channel, n=n, num_heads=1)

        # 标准 FC 注意力 (v1)
        if version == "v1":
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel, bias=False),
                nn.Sigmoid(),
            )

        # 增强型 FC 注意力 + BatchNorm + Dropout (v2)
        elif version == "v2":
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction, bias=True),
                nn.BatchNorm1d(channel // reduction),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(channel // reduction, channel, bias=True),
                nn.Sigmoid(),
            )

        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Args:
            x: [batch, channel, length]

        Returns:
            output: [batch, channel, length] - 注意力加权后的输出
        """
        n, c, l = x.shape

        # 自适应池化到 DCT 长度
        if l != self.dct_length:
            x_pooled = torch.nn.functional.adaptive_avg_pool1d(x, self.dct_length)
        else:
            x_pooled = x

        # 通过自适应 DCT 层提取频率特征
        # 输出: [batch, channel] - 每个通道的自适应频率特征
        y = self.dct_layer(x_pooled)

        # print("y shape:", y.shape)

        # 生成通道注意力权重
        y = self.fc(y).view(n, c, 1)  # [batch, channel, 1]

        # 应用注意力权重
        return x * y.expand_as(x)


class Self_onn_fcanet_ada2(nn.Module):
    """
    Self-ONN + FcaNet + 自适应频率选择 Model v2
    1. ✓ 自适应频率分量选择 (基于可训练权重)
    2. ✓ 改进的 FilterSelector 初始化 (Xavier + 随机扰动)
    3. ✓ Softmax 归一化权重，确保多样性
    4. ✓ 更好的梯度流和权重更新机制
    5. ✓ 支持增强型 FC 层 (v2: BatchNorm + Dropout)
    """

    def __init__(self, attention_version="v2"):
        super(Self_onn_fcanet_ada2, self).__init__()

        self.onn1 = nn.Sequential(
            SelfONN1d(in_channels=2, out_channels=32, kernel_size=15, padding=7, q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            SelfONN1d(in_channels=32, out_channels=32, kernel_size=11, padding=5, q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.fcanet1 = MultiSpectralAttentionLayer(
            channel=32, dct_length=1250, base_length=19, reduction=16, n=16, version=attention_version
        )

        self.onn2 = nn.Sequential(
            SelfONN1d(in_channels=32, out_channels=32, kernel_size=7, padding=3, q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            SelfONN1d(in_channels=32, out_channels=32, kernel_size=5, padding=2, q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.fcanet2 = MultiSpectralAttentionLayer(
            channel=32, dct_length=313, base_length=19, reduction=16, n=16, version=attention_version
        )

        self.onn3 = nn.Sequential(
            SelfONN1d(in_channels=32, out_channels=32, kernel_size=3, padding=1, q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            SelfONN1d(in_channels=32, out_channels=32, kernel_size=3, padding=1, q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.fcanet3 = MultiSpectralAttentionLayer(
            channel=32, dct_length=78, base_length=19, reduction=16, n=16, version=attention_version
        )

        self.onn4 = nn.Sequential(
            SelfONN1d(in_channels=32, out_channels=32, kernel_size=3, padding=1, q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            SelfONN1d(in_channels=32, out_channels=32, kernel_size=3, padding=1, q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.fcanet4 = MultiSpectralAttentionLayer(
            channel=32, dct_length=19, base_length=19, reduction=16, n=16, version=attention_version
        )

        self.classifier = nn.Sequential(
            nn.Linear(608, 64), nn.GELU(), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x1 = self.onn1(x)
        x1 = self.fcanet1(x1)

        x2 = self.onn2(x1)
        x2 = self.fcanet2(x2)

        x3 = self.onn3(x2)
        x3 = self.fcanet3(x3)

        x4 = self.onn4(x3)
        x4 = self.fcanet4(x4)

        x = self.flatten(x4)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = Self_onn_fcanet_ada2()
    summary(model, (1, 2, 5000))
