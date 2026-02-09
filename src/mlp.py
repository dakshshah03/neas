import torch
import torch.nn as nn

from encoder import get_encoder


class AttenuationActivation(nn.Module):
    """Custom activation function α·σ(x)+β for attenuation MLP output"""
    def __init__(self, alpha=1.0, beta=0.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, x):
        return self.alpha * torch.sigmoid(x) + self.beta


class MLPBlock(nn.Module):
    """Basic MLP block with configurable layers"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
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


class SDFNetwork(nn.Module):
    """SDF Network that outputs signed distance and feature vector
    
    Architecture follows paper specifications:
    - Frequency encoding: 6 layers, 256 hidden units
    - Hash encoding: 2 layers, 64 hidden units
    """
    def __init__(self, 
                 encoding='frequency',
                 input_dim=3,
                 feature_dim=8,
                 hidden_dim=256,
                 num_layers=6,
                 multires=6,
                 num_levels=14,
                 level_dim=2,
                 base_resolution=16,
                 log2_hashmap_size=19):
        super().__init__()
        
        self.encoding = encoding
        self.feature_dim = feature_dim
        
        # Get encoder
        self.encoder = get_encoder(
            encoding=encoding,
            input_dim=input_dim,
            multires=multires,
            num_levels=num_levels,
            level_dim=level_dim,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size
        )
        
        # MLP: outputs (1 distance + K feature dimensions)
        self.mlp = MLPBlock(
            input_dim=self.encoder.output_dim,
            hidden_dim=hidden_dim,
            output_dim=1 + feature_dim,
            num_layers=num_layers
        )
    
    def forward(self, x, tau=None):
        """
        Args:
            x: [N, 3] input positions
            tau: optional parameter for frequency regularization (only for frequency encoding)
        
        Returns:
            distances: [N] signed distances
            features: [N, K] feature vectors
        """
        # Encode input
        if self.encoding == 'frequency' and tau is not None:
            encoded = self.encoder(x, tau=tau)
        elif self.encoding == 'hashgrid':
            # Hash encoding expects inputs in range [-1, 1]
            encoded = self.encoder(x, size=1)
        else:
            encoded = self.encoder(x)
        
        # MLP forward
        output = self.mlp(encoded)
        
        # Split output
        distances = output[:, 0]
        features = output[:, 1:1+self.feature_dim] # maybe seperate heads so diff bias?
        
        return distances, features


class AttenuationNetwork(nn.Module):
    """Attenuation Network that outputs attenuation coefficient
    
    Architecture follows paper specifications:
    - Frequency encoding: 3 hidden layers, 256 hidden units  
    - Hash encoding: 2 layers, 64 hidden units
    
    Takes feature vector as input (not position).
    """
    def __init__(self,
                 input_dim=8,
                 hidden_dim=256,
                 num_layers=4,  # 3 hidden + 1 output for frequency
                 output_dim=1,
                 alpha=1.0,
                 beta=0.0):
        super().__init__()
        
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        
        # Hidden layers (num_layers - 2)
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.output_activation = AttenuationActivation(alpha=alpha, beta=beta)
    
    def forward(self, x):
        """
        Args:
            x: [N, K] feature vectors from SDF network
        
        Returns:
            attenuation: [N, 1] attenuation coefficients in range [β, β+α]
        """
        x = self.network(x)
        x = self.output_activation(x)
        return x


def create_neas_model(encoding='frequency',
                     feature_dim=8,
                     alpha=1.0,
                     beta=0.0,
                     multires=6,
                     num_levels=14,
                     level_dim=2,
                     base_resolution=16,
                     log2_hashmap_size=19):
    """Create NeAS model (SDF + Attenuation networks)
    
    Args:
        encoding: 'frequency' or 'hashgrid'
        feature_dim: dimension of feature vector (default 8)
        alpha: alpha parameter for attenuation activation
        beta: beta parameter for attenuation activation
        multires: number of frequency bands for frequency encoding
        num_levels: number of levels for hash encoding
        level_dim: feature dimension per level for hash encoding
        base_resolution: base resolution for hash encoding
        log2_hashmap_size: log2 of hash table size
    
    Returns:
        sdf_model, att_model
    """
    
    # Set architecture based on encoding type (from paper Section 4.2)
    if encoding == 'frequency':
        sdf_hidden_dim = 256
        sdf_num_layers = 6
        att_hidden_dim = 256
        att_num_layers = 4  # 3 hidden + 1 output
    elif encoding == 'hashgrid':
        sdf_hidden_dim = 64
        sdf_num_layers = 2
        att_hidden_dim = 64
        att_num_layers = 2
    else:
        raise ValueError(f"Unknown encoding: {encoding}")
    
    # Create SDF network
    sdf_model = SDFNetwork(
        encoding=encoding,
        input_dim=3,
        feature_dim=feature_dim,
        hidden_dim=sdf_hidden_dim,
        num_layers=sdf_num_layers,
        multires=multires,
        num_levels=num_levels,
        level_dim=level_dim,
        base_resolution=base_resolution,
        log2_hashmap_size=log2_hashmap_size
    )
    
    att_model = AttenuationNetwork(
        input_dim=feature_dim,
        hidden_dim=att_hidden_dim,
        num_layers=att_num_layers,
        output_dim=1,
        alpha=alpha,
        beta=beta
    )
    
    return sdf_model, att_model