from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastonn.SelfONN import SelfONN1d
from torchinfo import summary



import numpy as np
import math

def get_freq_indices(method, length):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32',
                      'alt1', 'alt2', 'alt4', 'alt8', 'alt16', 'alt32']

    indices = list(range(length))
    
    if method.startswith('top'):
        num = int(method[3:])
        return indices[-num:] 
    elif method.startswith('low'):
        num = int(method[3:])
        return indices[:num] 
    elif method.startswith('alt'):
        num = int(method[3:])
        alt_indices = []
        left, right = 0, length - 1
        
        while len(alt_indices) < num:
            if left <= right:
                alt_indices.append(left)
                left += 1
            if len(alt_indices) < num and left <= right:
                alt_indices.append(right)
                right -= 1
                
        return alt_indices
    return []

class MultiSpectralDCTLayer(nn.Module):
   
    def __init__(self, length, mapper, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        self.num_freq = len(mapper)
        assert channel % self.num_freq == 0

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(length, mapper, channel))
        
    def forward(self, x):
        assert len(x.shape) == 3, 'x must been 3 dimensions, but got ' + str(len(x.shape))
        # n, c, l = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size, mapper, channel):
        dct_filter = torch.zeros(channel, tile_size)

        c_part = channel // len(mapper)

        for i, u in enumerate(mapper):
            for t in range(tile_size):
                dct_filter[i * c_part: (i+1)*c_part, t] = self.build_filter(t, u, tile_size)
                        
        return dct_filter


class MultiSpectralAttentionLayer(nn.Module):
    def __init__(self, channel, dct_length, reduction=16, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_length = dct_length

        mapper_x = get_freq_indices(freq_sel_method, dct_length)
        # print(f"mapper_x: {mapper_x}")
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_length // len(mapper_x)) for temp_x in mapper_x] 
        

        self.dct_layer = MultiSpectralDCTLayer(dct_length, mapper_x, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

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
    


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastonn.SelfONN import SelfONN1d
from torchinfo import summary
class self_onn_fcanet(nn.Module):
    def __init__(self):
        super(self_onn_fcanet, self).__init__()
        self.onn1 = nn.Sequential(

            SelfONN1d(in_channels=2, out_channels=32, kernel_size=9, padding=4,q=3),
            
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            SelfONN1d(in_channels=32, out_channels=32, kernel_size=7, padding=3,q=3),
           
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # FECAM(2500),
            MultiSpectralAttentionLayer(32, dct_length=2500, freq_sel_method='low16')  
        )
        self.onn2 = nn.Sequential(
            SelfONN1d(in_channels=32, out_channels=32, kernel_size=5, padding=2,q=3),
            
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            SelfONN1d(in_channels=32, out_channels=32, kernel_size=3, padding=1,q=3),
           
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # FECAM(625),
            MultiSpectralAttentionLayer(32, dct_length=625, freq_sel_method='low16')  
        )

        self.onn3 = nn.Sequential(
            SelfONN1d(in_channels=32, out_channels=32, kernel_size=3, padding=1,q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            #  FECAM(312),
            SelfONN1d(in_channels=32, out_channels=32, kernel_size=3, padding=1,q=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # FECAM(156),
            MultiSpectralAttentionLayer(32, dct_length=156, freq_sel_method='low16')  
        )
        self.onn4 = nn.Sequential(
            SelfONN1d(in_channels=32, out_channels=64, kernel_size=3, padding=1,q=4),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # FECAM(78),
            SelfONN1d(in_channels=64, out_channels=64, kernel_size=3, padding=1,q=4),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            #   FECAM(39),
            MultiSpectralAttentionLayer(64, dct_length=39, freq_sel_method='low16')  
        )
        self.onn5 = nn.Sequential(
            SelfONN1d(in_channels=64, out_channels=64, kernel_size=3, padding=1,q=4),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            MultiSpectralAttentionLayer(64, dct_length=19, freq_sel_method='low16') 
        )
        self.classifier = nn.Sequential(
            nn.Linear(1216, 64),
            nn.GELU(),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.flatten = nn.Flatten()


    def forward(self, x):   # torch.Size([10, 2, 10000])
        x1 = self.onn1(x)
      

        x2 = self.onn2(x1)
        
        x3 = self.onn3(x2)
        

        x4 = self.onn4(x3)

        x5 = self.onn5(x4)

        x = self.flatten(x5)
        x = self.classifier(x)


        return x
if __name__ == '__main__':
    model = self_onn_fcanet()
    summary(model, input_size=(1, 2, 10000))