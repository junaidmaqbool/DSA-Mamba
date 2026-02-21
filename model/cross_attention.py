import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """
    Cross-Attention module that computes attention between query and key-value pairs.
    Handles both 3D (B, N, C) and 4D (B, H, W, C) tensor inputs with variable dimensions.
    
    Args:
        dim: Output dimension (also used for query if key_dim/value_dim not specified)
        key_dim: Internal key dimension (default: dim)
        value_dim: Internal value dimension (default: dim) 
        num_heads: Number of attention heads (default: 8)
        attn_drop: Attention dropout rate (default: 0.)
        proj_drop: Projection dropout rate (default: 0.)
    """
    
    def __init__(self, dim, key_dim=None, value_dim=None, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.key_dim = key_dim if key_dim is not None else dim
        self.value_dim = value_dim if value_dim is not None else dim
        
        # Ensure dimensions are compatible with num_heads
        assert self.key_dim % num_heads == 0, f"key_dim ({self.key_dim}) must be divisible by num_heads ({num_heads})"
        assert self.value_dim % num_heads == 0, f"value_dim ({self.value_dim}) must be divisible by num_heads ({num_heads})"
        
        self.head_dim_k = self.key_dim // num_heads
        self.head_dim_v = self.value_dim // num_heads
        self.scale = self.head_dim_k ** -0.5
        
        # We'll initialize projections lazily on first use
        # Store the layer definitions but not the actual layers yet
        self._q_proj = None
        self._kv_in_proj = None
        self._out_proj = None
        self._last_q_dim = None
        self._last_kv_dim = None
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def _ensure_projections_initialized(self, q_dim, kv_dim):
        """Initialize projection layers on first forward pass."""
        if self._q_proj is None or self._last_q_dim != q_dim or self._last_kv_dim != kv_dim:
            self._q_proj = nn.Linear(q_dim, self.key_dim, bias=True).to(self._get_device())
            self._kv_in_proj = nn.Linear(kv_dim, self.key_dim + self.value_dim, bias=True).to(self._get_device())
            self._out_proj = nn.Linear(self.value_dim, self.dim, bias=True).to(self._get_device())
            self._last_q_dim = q_dim
            self._last_kv_dim = kv_dim
    
    def _get_device(self):
        """Get device - fallback to CPU if no parameters registered yet."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')
    
    def forward(self, q, kv):
        """
        Args:
            q: Query input tensor of shape (B, N, C) or (B, H, W, C)
            kv: Key-value input tensor of shape (B, N, C) or (B, H, W, C), can have different C
        
        Returns:
            Output tensor of shape matching q
        """
        # Handle 4D (B, H, W, C) tensors by flattening spatial dimensions
        is_4d = q.dim() == 4
        original_shape = None
        if is_4d:
            B, H, W, C = q.shape
            original_shape = (B, H, W)
            q = q.reshape(B, H*W, C)
            kv = kv.reshape(kv.shape[0], -1, kv.shape[-1])
        
        B, N_q, C_q = q.shape
        B_kv, N_kv, C_kv = kv.shape
        
        # Initialize projections on first use
        self._ensure_projections_initialized(C_q, C_kv)
        
        # Project query
        q_proj = self._q_proj(q)  # (B, N_q, key_dim)
        q_proj = q_proj.reshape(B, N_q, self.num_heads, self.head_dim_k).permute(0, 2, 1, 3)  # (B, num_heads, N_q, head_dim_k)
        
        # Project kv
        kv_proj = self._kv_in_proj(kv)  # (B, N_kv, key_dim + value_dim)
        k_proj, v_proj = kv_proj.chunk(2, dim=-1)  # Split into k and v
        
        k_proj = k_proj.reshape(B_kv, N_kv, self.num_heads, self.head_dim_k).permute(0, 2, 1, 3)  # (B, num_heads, N_kv, head_dim_k)
        v_proj = v_proj.reshape(B_kv, N_kv, self.num_heads, self.head_dim_v).permute(0, 2, 1, 3)  # (B, num_heads, N_kv, head_dim_v)
        
        # Compute attention scores
        attn = (q_proj @ k_proj.transpose(-2, -1)) * self.scale  # (B, num_heads, N_q, N_kv)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = attn @ v_proj  # (B, num_heads, N_q, head_dim_v)
        x = x.transpose(1, 2).reshape(B, N_q, self.value_dim)  # (B, N_q, value_dim)
        x = self._out_proj(x)  # (B, N_q, dim)
        x = self.proj_drop(x)
        
        # Reshape back to 4D if input was 4D
        if is_4d:
            x = x.reshape(*original_shape, -1)
        
        return x





