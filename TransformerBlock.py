import torch
import torch.nn as nn
from MultiHeadSelfAttention import MultiHeadSelfAttention
from RMSNorm import RMSNorm
from SwiGLU import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int = 1024, theta: float = 10000.0, dropout: float = 0.0, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Args:
            d_model: int, dimensionality of the Transformer block inputs
            num_heads: int, number of heads to use in multi-head self-attention
            d_ff: int, dimensionality of the position-wise feed-forward inner layer
            max_seq_len: int, maximum sequence length for RoPE precomputation
            theta: float, RoPE parameter
            dropout: float, residual dropout applied to each sublayer output (0 = off)
            device: torch.device | None, device to store parameters on
            dtype: torch.dtype | None, data type of parameters
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # First sublayer: MultiHeadSelfAttention with RMSNorm
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.mha = MultiHeadSelfAttention(d_model, num_heads, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype)

        # Second sublayer: SwiGLU with RMSNorm
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff=d_ff, device=device, dtype=dtype)

        # Residual dropout (no parameters, so old checkpoints still load)
        self.drop = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor, input tensor of shape (..., seq_len, d_model)
            token_positions: torch.Tensor | None, tensor of shape (..., seq_len) with token positions
        
        Returns:
            torch.Tensor, output tensor of shape (..., seq_len, d_model)
        """
        output1 = x + self.drop(self.mha(self.ln1(x), token_positions))
        return output1 + self.drop(self.ffn(self.ln2(output1)))