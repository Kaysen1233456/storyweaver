import torch
import torch.nn as nn
import math
from RoPE import RotaryPositionalEmbedding
from function import softmax, scaled_dot_product_attention

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float = 10000.0, max_seq_len: int = 1024, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Args:
            d_model: int, dimensionality of the Transformer block inputs
            num_heads: int, number of heads to use in multi-head self-attention
            theta: float, RoPE parameter
            max_seq_len: int, maximum sequence length for RoPE precomputation
            device: torch.device | None, device to store parameters on
            dtype: torch.dtype | None, data type of parameters
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = self.d_v = d_model // num_heads

        # Projection layers
        self.w_q = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.w_k = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.w_v = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.w_o = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)

        # RoPE for query and key
        self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)

        # Initialize weights
        std = math.sqrt(2.0 / (d_model + d_model))
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3*std, b=3*std)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor, input tensor of shape (..., sequence_length, d_model)
        
        Returns:
            torch.Tensor, output tensor of shape (..., sequence_length, d_model)
        """
        *batch_dims, seq_len, d_model = x.shape

        # Project inputs to Q, K, V
        q = self.w_q(x)  # Shape: (..., seq_len, d_model)
        k = self.w_k(x)  # Shape: (..., seq_len, d_model)
        v = self.w_v(x)  # Shape: (..., seq_len, d_model)

        # Reshape for multi-head: (..., seq_len, num_heads, d_k)
        q = q.view(*batch_dims, seq_len, self.num_heads, self.d_k)
        k = k.view(*batch_dims, seq_len, self.num_heads, self.d_k)
        v = v.view(*batch_dims, seq_len, self.num_heads, self.d_k)

        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device).expand(*batch_dims, -1)

        # Apply RoPE to query and key for each head
        q_list = []
        k_list = []
        
        for head_idx in range(self.num_heads):
            q_head = q[..., head_idx, :]  # Shape: (..., seq_len, d_k)
            k_head = k[..., head_idx, :]  # Shape: (..., seq_len, d_k)
            
            q_rope = self.rope(q_head, token_positions)
            k_rope = self.rope(k_head, token_positions)
            
            q_list.append(q_rope)
            k_list.append(k_rope)

        # Stack back to multi-head format
        q = torch.stack(q_list, dim=-2)  # Shape: (..., seq_len, num_heads, d_k)
        k = torch.stack(k_list, dim=-2)  # Shape: (..., seq_len, num_heads, d_k)

        # Transpose to (... num_heads, seq_len, d_k) for attention calculation
        q = q.transpose(-3, -2)  # Shape: (..., num_heads, seq_len, d_k)
        k = k.transpose(-3, -2)  # Shape: (..., num_heads, seq_len, d_k)
        v = v.transpose(-3, -2)  # Shape: (..., num_heads, seq_len, d_k)

        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1) == 1
        causal_mask = causal_mask.logical_not()

        # Expand the mask to match mult-head dimensions
        if len(batch_dims) > 0:
            causal_mask = causal_mask.expand(*batch_dims, self.num_heads, -1, -1)
        else:
            causal_mask = causal_mask.expand(self.num_heads, -1, -1)

        # Apply scaled dot-product attention for each head
        output = scaled_dot_product_attention(q, k, v, causal_mask) # Output shape: (..., num_heads, seq_len, d_k)

        # Transpose to (..., seq_len, num_heads, d_k) then merge the heads
        output = output.transpose(-3, -2)  # Shape: (..., seq_len, num_heads, d_k)
        output = output.contiguous().view(*batch_dims, seq_len, self.d_model)

        return self.w_o(output)