import math
import torch
import torch.nn as nn
import numpy as np

from typing import Callable, Tuple

from ..encoder.freqencoder import FreqEncoder
from ..encoder.hashencoder import HashEncoder


def _make_hash_coarsefine_mask(tau, num_levels, level_dim, device, dtype):
    """Build the coarse-to-fine annealing mask for hash-encoded features.

    Replaces the per-call Python loop used in the original code with a
    fully-vectorised torch operation, eliminating repeated list construction
    and eager Python overhead on every forward pass.

    Weight per level k:  w(k) = clamp01(sigmoid-step annealing of tau - k)
        w = (1 - cos(clamp(tau - k, 0, 1) * π)) / 2
    This matches the original logic exactly:
        diff < 0  → w = 0    (level not yet unlocked)
        0 ≤ diff < 1 → cosine ramp
        diff ≥ 1  → w = 1    (level fully active)
    Each scalar weight is then broadcast over its ``level_dim`` features via
    repeat_interleave, giving a 1-D mask of length num_levels * level_dim.
    """
    k = torch.arange(num_levels, device=device, dtype=dtype)
    diff = torch.clamp(tau - k, 0.0, 1.0)
    w = (1.0 - torch.cos(diff * math.pi)) / 2.0   # shape: [num_levels]
    return w.repeat_interleave(level_dim)           # shape: [num_levels * level_dim]


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
                mask = _make_hash_coarsefine_mask(
                    tau, self.num_levels, self.level_dim,
                    encoded.device, encoded.dtype
                )
                encoded = encoded * mask
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
    return SDFMLPWrapper(encoder, mlp, feature_dim, encoding_type='hash', num_levels=num_levels, level_dim=level_dim)

def att_freq_mlp(input_dim=8, output_dim=1, alpha=3.4, beta=0.1):
    """Attenuation MLP for frequency encoding mode, custom activation function.
    
    The input is the feature vector f from the SDF network (paper Section III-A3),
    NOT spatial coordinates. No encoder is needed.
    
    Paper: "three hidden layers, with a size of 256" -> 4 total linear layers.
    
    Args:
        input_dim: Feature dimension K from SDF network
        alpha, beta: Activation parameters ensuring output in [beta, alpha+beta]
    """
    mlp = MLPBlock(input_dim, 256, output_dim, 4)  # 4 layers: input + 2 hidden + output = 3 hidden layers
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


class SDFMLPWrapperKM(nn.Module):
    """Wrapper for K-material SDF MLP that returns K distances and features.

    Handles an arbitrary number of materials K >= 2.

    Returns:
        distances: List of K tensors, each [B] — signed distance for material i
        features: [B, feature_dim] — feature vector for attenuation network
    """
    def __init__(self, encoder, mlp, num_materials, feature_dim, encoding_type='freq', num_levels=None, level_dim=None):
        super().__init__()
        self.encoder = encoder
        self.mlp = mlp
        self.num_materials = num_materials
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
                mask = _make_hash_coarsefine_mask(
                    tau, self.num_levels, self.level_dim,
                    encoded.device, encoded.dtype
                )
                encoded = encoded * mask
        else:
            encoded = self.encoder(x)

        output = self.mlp(encoded)
        K = self.num_materials
        distances = [output[:, i] for i in range(K)]
        features = output[:, K:K+self.feature_dim]
        return distances, features


class SharedAttenuationMLP(nn.Module):
    """Shared attenuation backbone with K material output heads.

    Merges independent attenuation MLPs into a single network with a shared
    backbone and separate output heads per material.  Each head consists of a
    single hidden layer responsible for predicting the raw attenuation μ̄_i.

    Architecture:
        features → [shared backbone] → shared_repr
        shared_repr → head_1 → μ̄_1   (with CustomActivation)
        shared_repr → head_2 → μ̄_2   (with CustomActivation)
        ...
        shared_repr → head_K → μ̄_K   (with CustomActivation)
    """
    def __init__(self, input_dim, hidden_dim, num_shared_layers, material_activations):
        """
        Args:
            input_dim: Feature dimension from SDF network
            hidden_dim: Hidden dimension for shared backbone and heads
            num_shared_layers: Number of hidden layers in shared backbone
            material_activations: List of (alpha, beta) tuples for each material
        """
        super().__init__()

        # Shared backbone
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_shared_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        self.backbone = nn.Sequential(*layers)

        # K output heads, each with 1 hidden layer + CustomActivation output
        self.heads = nn.ModuleList()
        for alpha, beta in material_activations:
            head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
                CustomActivation(alpha, beta)
            )
            self.heads.append(head)

    @property
    def num_materials(self):
        return len(self.heads)

    def forward(self, features):
        """Forward pass through shared backbone then per-material heads.

        Args:
            features: [B, input_dim] feature vectors from SDF network
        Returns:
            List of K tensors, each [B, 1], raw attenuation values per material
        """
        shared = self.backbone(features)
        return [head(shared) for head in self.heads]


def nested_material_selector(boundary_values, raw_attenuations):
    """Nested K-material priority selector for soft material composition.

    Computes:
        μ(x) = Σ_i  μ̄_i · Ω(d_i, s) · Π_{j>i} (1 − Ω(d_j, s))

    Materials are ordered from outermost (i=0, e.g. skin) to innermost
    (i=K-1, e.g. bone).  Each material's contribution is weighted by its
    own SBF value and the joint probability of NOT being inside any
    deeper/denser material.  For the innermost material the product is 1.

    Args:
        boundary_values: List of K SBF tensors, each [B]
        raw_attenuations: List of K raw attenuation tensors, each [B, 1]
    Returns:
        att_coeff: [B] total attenuation coefficient
    """
    K = len(boundary_values)
    att_coeff = torch.zeros_like(boundary_values[0])
    for i in range(K):
        w_i = boundary_values[i]
        for j in range(i + 1, K):
            w_i = w_i * (1.0 - boundary_values[j])
        att_coeff = att_coeff + raw_attenuations[i].squeeze(-1) * w_i
    return att_coeff


def sdf_freq_mlp_km(input_dim=3, num_materials=2, feature_dim=8, multires=6):
    """K-Material SDF MLP with frequency encoding.

    Returns (distances: list of K [B], features: [B, feature_dim]).
    """
    encoder = FreqEncoder(input_dim=input_dim, max_freq_log2=multires-1, N_freqs=multires,
                          log_sampling=True, include_input=True)
    mlp = MLPBlock(encoder.output_dim, 256, num_materials + feature_dim, 6)
    return SDFMLPWrapperKM(encoder, mlp, num_materials, feature_dim, encoding_type='freq')


def sdf_hash_mlp_km(input_dim=3, num_materials=2, feature_dim=8, num_levels=14, level_dim=2,
                    base_resolution=16, log2_hashmap_size=19):
    """K-Material SDF MLP with hash encoding.

    Returns (distances: list of K [B], features: [B, feature_dim]).
    Uses smaller MLP (2 layers, 64 hidden) as per paper for hash encoding.
    """
    encoder = HashEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim,
                          base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size)
    mlp = MLPBlock(encoder.output_dim, 64, num_materials + feature_dim, 2)
    return SDFMLPWrapperKM(encoder, mlp, num_materials, feature_dim, encoding_type='hash',
                           num_levels=num_levels, level_dim=level_dim)


def shared_att_freq_mlp(input_dim=8, material_activations=None):
    """Shared attenuation MLP for frequency encoding mode.

    Backbone: 2 hidden layers (256).  Each head: 1 hidden layer (256) + output.
    """
    return SharedAttenuationMLP(input_dim, hidden_dim=256, num_shared_layers=2,
                                material_activations=material_activations)


def shared_att_hash_mlp(input_dim=8, material_activations=None):
    """Shared attenuation MLP for hash encoding mode.

    Backbone: 1 hidden layer (64).  Each head: 1 hidden layer (64) + output.
    """
    return SharedAttenuationMLP(input_dim, hidden_dim=64, num_shared_layers=1,
                                material_activations=material_activations)


def get_network(net_type, encoding_type='freq', num_materials=1):
    """Get network constructor based on type, encoding, and number of materials.
    
    Args:
        net_type: 'sdf' or 'att'
        encoding_type: 'freq' or 'hash'
        num_materials: 1 for 1M-NeAS, >=2 for KM-NeAS
    """
    if net_type == 'sdf':
        if num_materials == 1:
            if encoding_type == 'freq':
                return sdf_freq_mlp
            elif encoding_type == 'hash':
                return sdf_hash_mlp
        else:  # K >= 2
            if encoding_type == 'freq':
                return sdf_freq_mlp_km
            elif encoding_type == 'hash':
                return sdf_hash_mlp_km
    elif net_type == 'att':
        if num_materials == 1:
            if encoding_type == 'freq':
                return att_freq_mlp
            elif encoding_type == 'hash':
                return att_hash_mlp
        else:  # K >= 2
            if encoding_type == 'freq':
                return shared_att_freq_mlp
            elif encoding_type == 'hash':
                return shared_att_hash_mlp
    
    raise NotImplementedError(f"Unknown network type: {net_type} with encoding: {encoding_type} and num_materials: {num_materials}")
