import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Tuple


class FreqEncoder(nn.Module):
    """Frequency Encoder for positional encoding using sine and cosine functions.
    
    Maps input from R^input_dim to R^output_dim using frequency-based encoding.
    This is the standard positional encoding used in NeRF and related methods.
    """
    def __init__(self,
            input_dim: int,
            max_freq_log2: float = 5,
            N_freqs: int = 5,
            log_sampling: bool = True,
            include_input: bool = True,
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
        """Calculate frequency weights for coarse-to-fine training."""
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
        """
        Args:
            input: [..., input_dim] tensor of positions
            tau: optional parameter for coarse-to-fine frequency regularization
            
        Returns:
            [..., output_dim] encoded tensor
        """
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
