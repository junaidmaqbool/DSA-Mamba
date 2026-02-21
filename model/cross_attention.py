import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """
    Cross-Attention module that computes attention between query and key-value pairs.
    Handles both 3D (B, N, C) and 4D (B, H, W, C) tensor inputs.
    
    Args:
        dim: Dimension of the query input
        key_dim: Dimension of the key
        value_dim: Dimension of the value
        num_heads: Number of attention heads (default: 8)
        attn_drop: Attention dropout rate (default: 0.)
        proj_drop: Projection dropout rate (default: 0.)
    """
    
    def __init__(self, dim, key_dim, value_dim, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.scale = (key_dim // num_heads) ** -0.5
        
        # Linear projections for query, key, value
        self.q_proj = nn.Linear(dim, key_dim, bias=True)
        self.k_proj = nn.Linear(key_dim, key_dim, bias=True)
        self.v_proj = nn.Linear(value_dim, value_dim, bias=True)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(value_dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, q, kv):
        """
        Args:
            q: Query input tensor of shape (B, N, C) or (B, H, W, C)
            kv: Key-value input tensor of shape (B, N, C) or (B, H, W, C)
        
        Returns:
            Output tensor with same shape as input
        """
        # Handle 4D (B, H, W, C) tensors by flattening spatial dimensions
        is_4d = q.dim() == 4
        original_shape = None
        if is_4d:
            B, H, W, C = q.shape
            original_shape = (B, H, W)
            q = q.reshape(B, H*W, C)
            kv = kv.reshape(kv.shape[0], -1, kv.shape[-1])
        
        B, N, C = q.shape
        
        # Project query, key, value
        q = self.q_proj(q).reshape(B, N, self.num_heads, self.key_dim // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(kv).reshape(B, kv.shape[1], self.num_heads, self.key_dim // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(kv).reshape(B, kv.shape[1], self.num_heads, self.value_dim // self.num_heads).permute(0, 2, 1, 3)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.value_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Reshape back to 4D if input was 4D
        if is_4d:
            x = x.reshape(*original_shape, -1)
        
        return x

