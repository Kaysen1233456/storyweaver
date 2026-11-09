import torch
import torch.nn as nn
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        """
        Args:
            theta: float, Θ value for the RoPE
            d_k: int, dimension of query and key vectors
            max_seq_len: int, maximum sequence length that will be inputted
            device: torch.device | None, device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        position = torch.arange(max_seq_len, device=device, dtype=torch.float32).unsqueeze(1)
        k = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        theta_values = 1.0 / (theta ** (k / d_k))
        angles = position * theta_values  # Shape: (max_seq_len, d_k/2)

        # Compute sin and cos for each position
        self.register_buffer('sin', torch.sin(angles), persistent=False)
        self.register_buffer('cos', torch.cos(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor, input tensor of shape (..., seq_len, d_k)
            token_positions: torch.Tensor, tensor of shape (..., seq_len) specifying token positions
        
        Returns:
            torch.Tensor, RoPEd tensor of the same shape as input
        """
        seq_len = x.shape[-2]

        # Adjust token_positions to be within [0, max_seq_len-1]
        token_positions = torch.clamp(token_positions, 0, self.max_seq_len - 1).long()

        original_shape = token_positions.shape
        flat_positions = token_positions.flatten()
        sin = self.sin[flat_positions].view(*original_shape, -1)  # Shape: (..., seq_len, d_k/2)
        cos = self.cos[flat_positions].view(*original_shape, -1)  # Shape: (..., seq_len, d_k/2)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        output = torch.empty_like(x)
        output[..., 0::2] = rotated_even
        output[..., 1::2] = rotated_odd

        return output