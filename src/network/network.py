import torch
import torch.nn as nn
import numpy as np

from typing import Callable, Tuple

from ..encoder.freqencoder import FreqEncoder
from ..encoder.hashencoder import HashEncoder


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


class CustomActivation(nn.Module):
    """Custom activation function: alpha * sigmoid(x) + beta
    
    This ensures output is in range [beta, alpha + beta] and prevents
    zero attenuation inside surfaces (beta > 0).
    """
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, x):
        return self.alpha * torch.sigmoid(x) + self.beta


class SDFMLPWrapper(nn.Module):
    """Wrapper for SDF MLP that returns distances and features separately"""
    def __init__(self, encoder, mlp, feature_dim, encoding_type='freq'):
        super().__init__()
        self.encoder = encoder
        self.mlp = mlp
        self.feature_dim = feature_dim
        self.encoding_type = encoding_type
    
    def forward(self, x, tau=None):
        if self.encoding_type == 'freq':
            encoded = self.encoder(x, bound=tau)
        elif self.encoding_type == 'hash':
            encoded = self.encoder(x, size=1.0)
        else:
            encoded = self.encoder(x)
        
        output = self.mlp(encoded)
        distances = output[:, 0] 
        features = output[:, 1:1+self.feature_dim]
        return distances, features


def sdf_freq_mlp(input_dim=3, output_dim=1, feature_dim=8, multires=6):
    """SDF MLP with frequency encoding - returns (distances [B], features [B, K])"""
    encoder = FreqEncoder(input_dim=input_dim, max_freq_log2=multires-1, N_freqs=multires, log_sampling=True, include_input=True)
    mlp = MLPBlock(encoder.output_dim, 256, output_dim + feature_dim, 6)
    return SDFMLPWrapper(encoder, mlp, feature_dim, encoding_type='freq')

def sdf_hash_mlp(input_dim=3, output_dim=1, feature_dim=8, num_levels=14, level_dim=2, base_resolution=16, log2_hashmap_size=19):
    """SDF MLP with hash encoding - returns (distances [B], features [B, K])
    
    Uses smaller MLP (2 layers, 64 hidden) as per paper for hash encoding.
    """
    encoder = HashEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, 
                         base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size)
    mlp = MLPBlock(encoder.output_dim, 64, output_dim + feature_dim, 2)
    return SDFMLPWrapper(encoder, mlp, feature_dim, encoding_type='hash')

def att_freq_mlp(input_dim=3, output_dim=1, multires=6, alpha=3.4, beta=0.1):
    """Attenuation MLP with frequency encoding, custom activation function.
    
    Args:
        alpha, beta: Activation parameters ensuring output in [beta, alpha+beta]
    """
    encoder = FreqEncoder(input_dim=input_dim, max_freq_log2=multires-1, N_freqs=multires, log_sampling=True, include_input=True)
    mlp = MLPBlock(encoder.output_dim, 256, output_dim, 4)
    activation = CustomActivation(alpha, beta)
    return nn.Sequential(encoder, mlp, activation)

def att_hash_mlp(input_dim=8, output_dim=1, num_levels=14, level_dim=2, base_resolution=16, log2_hashmap_size=19, alpha=3.4, beta=0.1):
    """Attenuation MLP with hash encoding, custom activation function.
    
    Note: input_dim is feature_dim from SDF network, not spatial dim.
    Uses smaller MLP (2 layers, 64 hidden) as per paper for hash encoding.
    """
    mlp = MLPBlock(input_dim, 64, output_dim, 2)
    activation = CustomActivation(alpha, beta)
    return nn.Sequential(mlp, activation)


def get_network(net_type, encoding_type='freq'):
    """Get network constructor based on type and encoding.
    
    Args:
        net_type: 'sdf' or 'att'
        encoding_type: 'freq' or 'hash'
    """
    if net_type == 'sdf':
        if encoding_type == 'freq':
            return sdf_freq_mlp
        elif encoding_type == 'hash':
            return sdf_hash_mlp
    elif net_type == 'att':
        if encoding_type == 'freq':
            return att_freq_mlp
        elif encoding_type == 'hash':
            return att_hash_mlp
    
    raise NotImplementedError(f"Unknown network type: {net_type} with encoding: {encoding_type}")
