import torch
import numpy as np

# TODO: CLEAN UP CODE
# hparams:
s = 10 

# surface boundary function
sbf = lambda d: (torch.exp(-s*d))/(1 + torch.exp(-s*d))


