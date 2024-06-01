import torch
import torch.nn as nn
import numpy as np
from .spline import *
from .KANLayer import *


class MultKANLayer(nn.Module):

    def __init__(self, in_dim=3, out_dim_sum=2, out_dim_mult=2, num=5, k=3, noise_scale=0.1, scale_base=1.0, scale_sp=1.0, base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, device='cpu'):
        
        super(MultKANLayer, self).__init__()
        self.out_dim_sum = out_dim_sum
        self.out_dim_mult = out_dim_mult
        self.kanlayer = KANLayer(in_dim=in_dim, out_dim=out_dim_sum+2*out_dim_mult, num=num, k=k, noise_scale=noise_scale, scale_base=scale_base, scale_sp=scale_sp, base_fun=base_fun, grid_eps=grid_eps, grid_range=grid_range, sp_trainable=sp_trainable, sb_trainable=sb_trainable, device=device)
        self.device = device

    def forward(self, x):
        out_dim_sum = self.out_dim_sum
        y, preacts, postacts, postspline = self.kanlayer(x)
        y_mult = y[:,out_dim_sum::2] * y[:,out_dim_sum+1::2]
        y = torch.cat([y[:,:out_dim_sum], y_mult], dim=1)
        return y, preacts, postacts, postspline
