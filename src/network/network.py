import torch
import torch.nn as nn
import numpy as np

from typing import Callable, Tuple

# use the standalone FreqEncoder implementation
from ..encoder.freqencoder import FreqEncoder


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
        # the external encoder expects `bound` (coarse-to-fine tau); map the name
        encoded = self.encoder(x, bound=tau)
        output = self.mlp(encoded)
        distances = output[:, 0] 
        features = output[:, 1:1+self.feature_dim]
        return distances, features


def sdf_freq_mlp(input_dim=3, output_dim=1, feature_dim=8, multires=6):
    """SDF MLP with frequency encoding - returns (distances [B], features [B, K])"""
    encoder = FreqEncoder(input_dim=input_dim, max_freq_log2=multires-1, N_freqs=multires, log_sampling=True, include_input=True)
    mlp = MLPBlock(encoder.output_dim, 256, output_dim + feature_dim, 6)
    return SDFMLPWrapper(encoder, mlp, feature_dim)

def att_freq_mlp(input_dim=3, output_dim=1, multires=6):
    """Attenuation MLP with frequency encoding, 256 hidden size"""
    encoder = FreqEncoder(input_dim=input_dim, max_freq_log2=multires-1, N_freqs=multires, log_sampling=True, include_input=True)
    mlp = MLPBlock(encoder.output_dim, 256, output_dim, 4)
    return nn.Sequential(encoder, mlp)


def get_network(net_type):
    """Get network constructor based on type."""
    if net_type == 'sdf_mlp':
        return sdf_freq_mlp
    elif net_type == 'att_mlp':
        return att_freq_mlp
    else:
        raise NotImplementedError(f"Unknown network type: {net_type}")
