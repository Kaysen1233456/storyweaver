import torch
import torch.nn as nn
import math
from RoPE import RotaryPositionalEmbedding
from function import scaled_dot_product_attention

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

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_k = self.d_v = d_model // num_heads
        if self.d_k % 2 != 0:
            raise ValueError("head dimension must be even for RoPE")

        # Projection layers
        self.w_q = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.w_k = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.w_v = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.w_o = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)

        # RoPE for query and key
        self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device)),
            persistent=False,
        )

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
        *batch_dims, seq_len, _ = x.shape

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

        # Apply RoPE to query and key, broadcast over heads:
        # positions (..., seq_len) -> (..., seq_len, num_heads), pairing with (..., seq_len, num_heads, d_k)
        pos = token_positions.unsqueeze(-1).expand(*token_positions.shape, self.num_heads)
        q = self.rope(q, pos)
        k = self.rope(k, pos)

        # Transpose to (... num_heads, seq_len, d_k) for attention calculation
        q = q.transpose(-3, -2)  # Shape: (..., num_heads, seq_len, d_k)
        k = k.transpose(-3, -2)  # Shape: (..., num_heads, seq_len, d_k)
        v = v.transpose(-3, -2)  # Shape: (..., num_heads, seq_len, d_k)

        # Reuse the precomputed mask; its 2-D shape broadcasts over batch and heads
        causal_mask = self.causal_mask[:seq_len, :seq_len]

        # Apply scaled dot-product attention for each head
        output = scaled_dot_product_attention(q, k, v, causal_mask) # Output shape: (..., num_heads, seq_len, d_k)

        # Transpose to (..., seq_len, num_heads, d_k) then merge the heads
        output = output.transpose(-3, -2)  # Shape: (..., seq_len, num_heads, d_k)
        output = output.contiguous().view(*batch_dims, seq_len, self.d_model)

        return self.w_o(output)