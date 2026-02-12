import torch
import torch.nn as nn
import numpy as np

from typing import Callable, Tuple

from encoder.freqencoder import FreqEncoder

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
    def __init__(self, encoder, shared_mlp, distance_head, feature_head, feature_dim):
        super().__init__()
        self.encoder = encoder
        self.shared_mlp = shared_mlp
        self.distance_head = distance_head
        self.feature_head = feature_head
        self.feature_dim = feature_dim
    
    def forward(self, x, tau=None):
        # the external encoder expects `bound` (coarse-to-fine tau); map the name
        encoded = self.encoder(x, bound=tau)
        shared_features = self.shared_mlp(encoded)
        
        # Separate heads for distance and features
        distances = self.distance_head(shared_features).squeeze(-1)  # [B]
        features = self.feature_head(shared_features)  # [B, feature_dim]
        
        return distances, features


def sdf_freq_mlp(input_dim=3, output_dim=1, feature_dim=8, multires=6):
    """SDF MLP with frequency encoding - returns (distances [B], features [B, K])
    Uses separate heads for distance and features to decouple geometry and appearance learning.
    """
    encoder = FreqEncoder(input_dim=input_dim, max_freq_log2=multires-1, N_freqs=multires, log_sampling=True, include_input=True)
    
    # Shared backbone (all layers except the final output)
    shared_layers = []
    hidden_dim = 256
    num_layers = 6
    
    # First layer
    shared_layers.append(nn.Linear(encoder.output_dim, hidden_dim))
    shared_layers.append(nn.ReLU(inplace=True))
    
    # Hidden layers (all but the last)
    for _ in range(num_layers - 2):
        shared_layers.append(nn.Linear(hidden_dim, hidden_dim))
        shared_layers.append(nn.ReLU(inplace=True))
    
    shared_mlp = nn.Sequential(*shared_layers)
    
    # Separate output heads
    distance_head = nn.Linear(hidden_dim, output_dim)
    feature_head = nn.Linear(hidden_dim, feature_dim)
    
    return SDFMLPWrapper(encoder, shared_mlp, distance_head, feature_head, feature_dim)

def att_freq_mlp(input_dim=3, output_dim=1, multires=6):
    """Attenuation MLP with frequency encoding, 256 hidden size"""
    encoder = FreqEncoder(input_dim=input_dim, max_freq_log2=multires-1, N_freqs=multires, log_sampling=True, include_input=True)
    mlp = MLPBlock(encoder.output_dim, 256, output_dim, 4)
    return nn.Sequential(encoder, mlp)