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
    def __init__(self, encoder, mlp, feature_dim, encoding_type='freq', num_levels=None, level_dim=None):
        super().__init__()
        self.encoder = encoder
        self.mlp = mlp
        self.feature_dim = feature_dim
        self.encoding_type = encoding_type
        self.num_levels = num_levels
        self.level_dim = level_dim
    
    def forward(self, x, tau=None):
        if self.encoding_type == 'freq':
            encoded = self.encoder(x, bound=tau)
        elif self.encoding_type == 'hash':
            encoded = self.encoder(x, size=1.0)
            if tau is not None and self.num_levels is not None and self.level_dim is not None:
                import math
                weights = []
                for k in range(self.num_levels):
                    diff = tau - k
                    if diff < 0:
                        w = 0.0
                    elif diff < 1:
                        w = (1.0 - math.cos(diff * math.pi)) / 2.0
                    else:
                        w = 1.0
                    weights.extend([w] * self.level_dim)
                mask = torch.tensor(weights, device=encoded.device, dtype=encoded.dtype)
                encoded = encoded * mask
        else:
            encoded = self.encoder(x)
        
        output = self.mlp(encoded)
        distances = output[:, 0] 
        features = output[:, 1:1+self.feature_dim]
        return distances, features


class SDFMLPWrapper2M(nn.Module):
    """Wrapper for 2-material SDF MLP that returns TWO distances and features.
    
    Returns:
        d1 [B]: Signed distance for material 1 (e.g., skin-air surface)
        d2 [B]: Signed distance for material 2 (e.g., bone-muscle surface)
        features [B, K]: Feature vector for attenuation networks
    """
    def __init__(self, encoder, mlp, feature_dim, encoding_type='freq', num_levels=None, level_dim=None):
        super().__init__()
        self.encoder = encoder
        self.mlp = mlp
        self.feature_dim = feature_dim
        self.encoding_type = encoding_type
        self.num_levels = num_levels
        self.level_dim = level_dim
    
    def forward(self, x, tau=None):
        if self.encoding_type == 'freq':
            encoded = self.encoder(x, bound=tau)
        elif self.encoding_type == 'hash':
            encoded = self.encoder(x, size=1.0)
            if tau is not None and self.num_levels is not None and self.level_dim is not None:
                import math
                weights = []
                for k in range(self.num_levels):
                    diff = tau - k
                    if diff < 0:
                        w = 0.0
                    elif diff < 1:
                        w = (1.0 - math.cos(diff * math.pi)) / 2.0
                    else:
                        w = 1.0
                    weights.extend([w] * self.level_dim)
                mask = torch.tensor(weights, device=encoded.device, dtype=encoded.dtype)
                encoded = encoded * mask
        else:
            encoded = self.encoder(x)
        
        output = self.mlp(encoded)
        d1 = output[:, 0]  # First SDF (skin-air)
        d2 = output[:, 1]  # Second SDF (bone-muscle)
        features = output[:, 2:2+self.feature_dim]
        return d1, d2, features


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
    return SDFMLPWrapper(encoder, mlp, feature_dim, encoding_type='hash', num_levels=num_levels, level_dim=level_dim)

def att_freq_mlp(input_dim=8, output_dim=1, alpha=3.4, beta=0.1):
    """Attenuation MLP for frequency encoding mode, custom activation function.
    
    The input is the feature vector f from the SDF network (paper Section III-A3),
    NOT spatial coordinates. No encoder is needed.
    
    Paper: "three hidden layers, with a size of 256" -> 5 total layers.
    
    Args:
        input_dim: Feature dimension K from SDF network
        alpha, beta: Activation parameters ensuring output in [beta, alpha+beta]
    """
    mlp = MLPBlock(input_dim, 256, output_dim, 5)  # 5 layers: input + 3 hidden + output
    activation = CustomActivation(alpha, beta)
    return nn.Sequential(mlp, activation)

def att_hash_mlp(input_dim=8, output_dim=1, num_levels=14, level_dim=2, base_resolution=16, log2_hashmap_size=19, alpha=3.4, beta=0.1):
    """Attenuation MLP with hash encoding, custom activation function.
    
    Note: input_dim is feature_dim from SDF network, not spatial dim.
    Uses smaller MLP (2 layers, 64 hidden) as per paper for hash encoding.
    """
    mlp = MLPBlock(input_dim, 64, output_dim, 2)
    activation = CustomActivation(alpha, beta)
    return nn.Sequential(mlp, activation)


def sdf_freq_mlp_2m(input_dim=3, output_dim=2, feature_dim=8, multires=6):
    """2-Material SDF MLP with frequency encoding.
    
    Returns (d1 [B], d2 [B], features [B, K]) where:
    - d1: Signed distance for material 1 (e.g., skin-air surface)
    - d2: Signed distance for material 2 (e.g., bone-muscle surface)
    - features: Feature vector for attenuation networks
    """
    encoder = FreqEncoder(input_dim=input_dim, max_freq_log2=multires-1, N_freqs=multires, log_sampling=True, include_input=True)
    mlp = MLPBlock(encoder.output_dim, 256, output_dim + feature_dim, 6)
    return SDFMLPWrapper2M(encoder, mlp, feature_dim, encoding_type='freq')


def sdf_hash_mlp_2m(input_dim=3, output_dim=2, feature_dim=8, num_levels=14, level_dim=2, base_resolution=16, log2_hashmap_size=19):
    """2-Material SDF MLP with hash encoding.
    
    Returns (d1 [B], d2 [B], features [B, K]) where:
    - d1: Signed distance for material 1 (e.g., skin-air surface)
    - d2: Signed distance for material 2 (e.g., bone-muscle surface)  
    - features: Feature vector for attenuation networks
    
    Uses smaller MLP (2 layers, 64 hidden) as per paper for hash encoding.
    """
    encoder = HashEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, 
                         base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size)
    mlp = MLPBlock(encoder.output_dim, 64, output_dim + feature_dim, 2)
    return SDFMLPWrapper2M(encoder, mlp, feature_dim, encoding_type='hash', num_levels=num_levels, level_dim=level_dim)


def selector_function(d2, mu1, mu2):
    """Selector function Λ(d2, μ1, μ2) for choosing between two attenuation coefficients.
    
    As described in the paper:
    - If d2 < 0 (point is inside material 2, e.g., bone): use μ2
    - If d2 >= 0 (point is outside material 2): use μ1
    
    Args:
        d2: Signed distance for material 2 [B, ...]
        mu1: Attenuation coefficient for material 1 [B, ...]
        mu2: Attenuation coefficient for material 2 [B, ...]
    
    Returns:
        mu: Selected attenuation coefficient [B, ...]
    """
    # Use torch.where for differentiable selection
    # d2 < 0 means inside material 2 (bone), use mu2
    # d2 >= 0 means outside material 2 (muscle/soft tissue), use mu1
    mu = torch.where(d2 < 0, mu2, mu1)
    return mu


def get_network(net_type, encoding_type='freq', num_materials=1):
    """Get network constructor based on type, encoding, and number of materials.
    
    Args:
        net_type: 'sdf' or 'att'
        encoding_type: 'freq' or 'hash'
        num_materials: 1 for 1M-NeAS, 2 for 2M-NeAS
    """
    if net_type == 'sdf':
        if num_materials == 1:
            if encoding_type == 'freq':
                return sdf_freq_mlp
            elif encoding_type == 'hash':
                return sdf_hash_mlp
        elif num_materials == 2:
            if encoding_type == 'freq':
                return sdf_freq_mlp_2m
            elif encoding_type == 'hash':
                return sdf_hash_mlp_2m
    elif net_type == 'att':
        if encoding_type == 'freq':
            return att_freq_mlp
        elif encoding_type == 'hash':
            return att_hash_mlp
    
    raise NotImplementedError(f"Unknown network type: {net_type} with encoding: {encoding_type} and num_materials: {num_materials}")
