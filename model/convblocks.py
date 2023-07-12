import torch
import torch.nn as nn
import torch.nn.functional as F

from model.activation.activation import Swish

class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers=2, scale_factor=2, groups=32, dropout=0.2):
        super(Conv_Block, self).__init__()
        self.n_layers = n_layers

        clist = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)]
        for _ in range(n_layers-1):
            clist.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False))

        self.conv_layers = nn.ModuleList(clist)
        self.pool = nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = Swish()
        

    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)
            x = self.act(self.norm(x))
        x = self.pool(x)    
        return x


class SkipConnection(nn.Module):
    def __init__(self, channels, groups=32, scale_factor=2):
        super(SkipConnection, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor)
        self.norm = nn.GroupNorm(groups, channels)
        self.act = Swish()

    def forward(self, x: torch.Tensor):
        x_e = self.pool(x)
        x_e = self.act(self.norm(x_e))
        return x_e, x



class CombineConnection(nn.Module):
    def __init__(self, channels, groups=32, scale_factor=2):
        super(CombineConnection, self).__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.norm = nn.GroupNorm(groups, channels)
        self.act = Swish()

    def forward(self, x, x_e):
        x = self.up(x)
        x = self.act(self.norm(x))
        x = torch.add(x, x_e)
        return x
