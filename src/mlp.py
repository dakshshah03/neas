import torch
import torch.nn as nn
import numpy as np

from typing import Callable, Tuple

class FreqEncoder(nn.Module):
    def __init__(self,
            input_dim: int,
            max_freq_log2: float=5,
            N_freqs: int = 5,
            log_sampling: bool=True,
            include_input: bool=True,
            periodic_fns: Tuple[Callable[[torch.Tensor], torch.Tensor], ...] = (torch.sin, torch.cos)):
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns
        self.N_freqs = N_freqs

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim

        self.output_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.register_buffer('freq_bands', freq_bands)
        
        self.tau = None

    def get_freq_weights(self, tau):
        weights = torch.zeros(self.N_freqs, device=self.freq_bands.device)
        
        for k in range(self.N_freqs):
            diff = tau - k
            if diff < 0:
                weights[k] = 0.0
            elif 0 <= diff < 1:
                weights[k] = (1 - torch.cos(torch.tensor(diff * np.pi))) / 2
            else:
                weights[k] = 1.0
        
        return weights

    def forward(self, input, tau=None):
        out = []
        if self.include_input:
            out.append(input)

        if tau is not None:
            weights = self.get_freq_weights(tau)
        else:
            weights = torch.ones(self.N_freqs, device=self.freq_bands.device)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            weight = weights[i]
            for p_fn in self.periodic_fns:
                out.append(weight * p_fn(input * freq))

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
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers):
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


class SDFMLPWrapper(nn.Module):
    """Wrapper for SDF MLP that returns distances and features separately"""
    def __init__(self, encoder, mlp, feature_dim):
        super().__init__()
        self.encoder = encoder
        self.mlp = mlp
        self.feature_dim = feature_dim
    
    def forward(self, x, tau=None):
        encoded = self.encoder(x, tau=tau)
        output = self.mlp(encoded)
        distances = output[:, 0] 
        features = output[:, 1:1+self.feature_dim]
        return distances, features


def sdf_freq_mlp(input_dim=3, output_dim=1, feature_dim=8):
    """SDF MLP with frequency encoding - returns (distances [B], features [B, K])"""
    encoder = FreqEncoder(input_dim=input_dim)
    mlp = MLPBlock(encoder.output_dim, 256, output_dim + feature_dim, 6)
    return SDFMLPWrapper(encoder, mlp, feature_dim)

def att_freq_mlp(input_dim=3, output_dim=1):
    """Attenuation MLP with frequency encoding, 256 hidden size"""
    encoder = FreqEncoder(input_dim=input_dim)
    mlp = MLPBlock(encoder.output_dim, 256, output_dim, 4)
    return nn.Sequential(encoder, mlp)



# def sdf_hash_mlp(input_dim=3, output_dim=1):
#     """SDF MLP with hash encoding - 2 layers, 64 hidden size"""
#     encoder = HashEncoder(input_dim=input_dim)
#     mlp = MLPBlock(encoder.out_dim, 64, output_dim, 2)
#     return nn.Sequential(encoder, mlp)


# def att_hash_mlp(input_dim=3, output_dim=1):
#     """Attenuation MLP with hash encoding - 2 layers, 64 hidden size"""
#     encoder = HashEncoder(input_dim=input_dim)
#     mlp = MLPBlock(encoder.out_dim, 64, output_dim, 2)
#     return nn.Sequential(encoder, mlp)