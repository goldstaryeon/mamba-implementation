import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from pscan import pscan

@dataclass
class MambaConfig:
    d_model: int #D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16 #N
    expand_factor: int = 2 #E
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False
    pscan: bool = True
    use_cuda: bool = False

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        
        self.config = config
        self.in_proj = nn.Linear(config.d_model, 2*config.d_inner, bias=config.bias)
        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                kernel_size=config.d_conv, bias=config.conv_bias,
                                groups=config.d_inner,
                                padding=config.d_conv-1)
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_

class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = 

class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config
        self.layers = nn.ModuleList([Residualblock])


