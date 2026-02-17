import torch
import torch.nn as nn

import math


class FreqEncoder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):

        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.N_freqs = N_freqs
        self.max_freq_log2 = max_freq_log2

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim

        self.output_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = freq_bands.numpy().tolist()

    def forward(self, input, bound=None):
        """Encode `input` with sin/cos frequency bands.

        Args:
            input: tensor [..., input_dim]
            bound: None or float. If provided, applies coarse-to-fine weighting w_k(bound)
                   to each frequency band k as described in the paper. If None, all bands
                   are fully enabled.
        """

        out = []
        if self.include_input:
            out.append(input)

        # build per-frequency weights based on `bound` (coarse-to-fine schedule)
        if bound is None:
            weights = [1.0] * self.N_freqs
        else:
            weights = []
            for k in range(self.N_freqs):
                diff = bound - k
                if diff < 0:
                    w = 0.0
                elif diff < 1:
                    # smooth transition when 0 <= diff < 1
                    w = (1.0 - math.cos(diff * math.pi)) / 2.0
                else:
                    w = 1.0
                weights.append(w)

        for i, freq in enumerate(self.freq_bands):
            w = weights[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq * math.pi) * w)

        out = torch.cat(out, dim=-1)
        return out