import torch
import torch.nn as nn
import numpy as np

from typing import Callable, Tuple

class FreqEncoder(nn.Module):
    def __init__(self,
            input_dim: int,
            max_freq_log2: float,
            N_freqs: int,
            log_sampling: bool=True,
            include_input: bool=True,
            periodic_fns: Tuple[Callable[[torch.Tensor], torch.Tensor], ...] = (torch.sin, torch.cos)):
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim

        self.output_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self,
                input,
                bound):
        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))

        out = torch.cat(out, dim=-1)
        
        return out


# TODO: copy hash encoder implementation from https://github.com/Ruyi-Zha/naf_cbct/tree/main/src/encoder/hashencoder and somehow use the cuda bindings


class MLPBlock(nn.Module):
    """
    Basic MLP block

    Args:
        nn (_type_): _description_
    """
    def __init__(self,
                 input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def sdf_freq_mlp(input_dim=3, output_dim=1):
    """SDF MLP with frequency encoding - 6 layers, 256 hidden size"""
    encoder = FrequencyEncoder(input_dim=input_dim)
    mlp = MLPBlock(encoder.out_dim, 256, output_dim, 6)
    return nn.Sequential(encoder, mlp)


def sdf_hash_mlp(input_dim=3, output_dim=1):
    """SDF MLP with hash encoding - 2 layers, 64 hidden size"""
    encoder = HashEncoder(input_dim=input_dim)
    mlp = MLPBlock(encoder.out_dim, 64, output_dim, 2)
    return nn.Sequential(encoder, mlp)


def att_freq_mlp(input_dim=3, output_dim=1):
    """Attenuation MLP with frequency encoding - 3 layers, 256 hidden size"""
    encoder = FrequencyEncoder(input_dim=input_dim)
    mlp = MLPBlock(encoder.out_dim, 256, output_dim, 3)
    return nn.Sequential(encoder, mlp)


def att_hash_mlp(input_dim=3, output_dim=1):
    """Attenuation MLP with hash encoding - 2 layers, 64 hidden size"""
    encoder = HashEncoder(input_dim=input_dim)
    mlp = MLPBlock(encoder.out_dim, 64, output_dim, 2)
    return nn.Sequential(encoder, mlp)