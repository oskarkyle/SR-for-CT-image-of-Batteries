import math
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F_torch

class SRResNet(nn.Module):
    def __init__(self, 
                 in_channel:int,
                 out_channel:int,
                 num_rcb:int,
                 upscale_factor:int
                 ) -> None:
        super().__init__()
        