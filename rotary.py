from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch import Tensor

def precompute_freqs_cis(head_dim: int, seq_len: int, device: str = 'cuda', theta: float = 10000.0):
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"

    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) 
    m = torch.arange(seq_len, device=device)
    freqs = torch.outer(m, theta).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, device: str = 'cuda'):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)