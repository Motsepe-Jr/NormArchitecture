

# import softmax, sqrt, concat, mean_pool
from torch.nn.functional import softmax
from math import sqrt
from torch import concat
from torch import Tensor

# Multi-Query Attention
# Lets assume we already have a pretrained model (MHA)

def mean_pool(tensor, num_groups, heads_per_group):
    batch_size, num_heads, seq_len, d = tensor.shape
       # I am using a view because I want to avoid copying the data
       # assumign the tesnor has continuguos memomey
    tensor = tensor.view(batch_size, num_groups, heads_per_group, seq_len, d)
                         # b, num_groups, heads_per_group, 512, 2048
                              # 4         # 2
    return tensor.mean(dim=2)  


def grouped_query_attention(Q, K_heads, V_heads, num_groups):

    # B, H, N, D
    
    num_heads = Q.shape[1] # B, H, N D
    heads_per_group = num_heads // 1

    grouped_K = mean_pool(K_heads, num_groups, heads_per_group)  # B, Num_groups, Heads_per_group, N, D
    grouped_V = mean_pool(V_heads, num_groups, heads_per_group) 

    attention_outputs = []
    
    for g in range(num_groups):

        # 

        Q_group = Q[:, g * heads_per_group : (g + 1) * heads_per_group] #  -> B, 2, N D
        K_group = grouped_K[:, g] # key head ( mean polled )   # B, 0, Heads_per_group, N, D
        V_group = grouped_V[:, g] # value head (mean polled)
     
        attn_scores = softmax(Q_group @ K_group.transpose(-2, -1) / sqrt(d_k)) #  B, 2, N D   @  B, 0, Heads_per_group, D, N  -> B, 0, Heads_per_group, N, N
        output = attn_scores @ V_group 

        attention_outputs.append(output)

    return concat(attention_outputs, dim=1) 


# Multi-Query Attention (MQA)
# MQA is just GQA with G = 1 (single key-value for all heads).
# The only difference is that we don’t divide the queries into multiple groups.

import torch.nn.functional as F
import torch

def multi_query_attention(Q, K, V):

    """
    K: Keys, shape [B, 1, M, D]  (Shared across all heads)
    V: Values, shape [B, 1, M, D] (Shared across all heads)
    """
   
    B, H, N, D = Q.shape  
    _, _, M, _ = K.shape  
    
    # Compute Attention Scores: (B, H, N, D) @ (B, 1, D, M) -> (B, H, N, M)

    # (B, H, N, D) @ (B, 1, D, M) 

    attn_scores = torch.matmul(Q, K.transpose(-1, -2))
    attn_probs = F.softmax(attn_scores, dim=-1)  
    
    Y = torch.matmul(attn_probs, V)
    
    return Y

from torch import nn


## Hybrid Attention Horizons.


# Method	        Number of Key-Value Heads	Memory Efficiency	        Diversity
# MHA	            H (one per head)	        High (expensive)	        High
# MQA	            1 (shared)	                Very High (low KV-cache)	Low # 
# GQA (G groups)	G (one per group)	        Medium	                    Medium



#  Hybrid Attention Horizons. We interleave local attention (Beltagy et al., 2020) with global attention layers. Local attention is trained with sliding windows, and reduces the complexity from O(length2) to O(length). We found that reducing attention horizon to 1024 on most attention layers does not have a significant impact on evaluation metrics, including the long context needle-in-haystack benchmark. 
#  In our production model, only 1 out of every 6 layers uses global attention.

class SlidingWindowAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, window_size):
        super(SlidingWindowAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        
       
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
      
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
    
        attn_output = []
        for i in range(seq_len):
            left_bound = max(0, i - self.window_size // 2)
            right_bound = min(seq_len, i + self.window_size // 2 + 1)
            
            q_i = q[:, :, i:i+1, :] 
            k_window = k[:, :, left_bound:right_bound, :]
            v_window = v[:, :, left_bound:right_bound, :]
            
            scores = torch.matmul(q_i, k_window.transpose(-1, -2)) / (self.head_dim ** 0.5)
            
            attn_weights = F.softmax(scores, dim=-1)
            
            attn_i = torch.matmul(attn_weights, v_window)
            attn_output.append(attn_i)
        
        attn_output = torch.cat(attn_output, dim=2)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        output = self.output(attn_output)
        
        return output



class InterleaveAttentionModel(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers, window_size, interleave_factor=6):
        super(InterleaveAttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.interleave_factor = interleave_factor  
        
      
        self.attention_layers = nn.ModuleList()
        for i in range(num_layers):
            if (i + 1) % interleave_factor == 0: 
                self.attention_layers.append(
                    nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=num_heads,
                        dim_feedforward=hidden_size * 4,
                        batch_first=True
                    )
                )
            else:
                self.attention_layers.append(
                    SlidingWindowAttention(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        window_size=window_size
                    )
                )
        
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x, global_indices=None):
  
        for i, layer in enumerate(self.attention_layers):
            if (i + 1) % self.interleave_factor == 0:
               
                x = layer(x)
            else:
                x = x + layer(x)  
                x = self.norm(x)
        
        return x








