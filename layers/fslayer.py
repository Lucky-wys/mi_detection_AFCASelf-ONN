import math
import torch
import torch.nn as nn


class FSLayer(nn.Module):
    def __init__(self, num_features):
        super(FSLayer, self).__init__()
        self.num_features = num_features
        self.weights = nn.ParameterList([])
        self.init_weights()

    def init_weights(self):
        for i in range(self.num_features):
            self.weights.append(nn.Parameter(torch.empty(1, 1), requires_grad=True))
            nn.init.kaiming_uniform_(self.weights[i], a=math.sqrt(5))

    def forward(self, x: list):
        for i in range(self.num_features):
            x[i] = x[i] * self.weights[i]
        return torch.cat(x, dim=1)
