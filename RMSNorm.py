import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Args:
            d_model: int, hidden dimension of the model
            eps: float, epsilon value for numerical stability (default: 1e-5)
            device: torch.device | None, device to store the parameters on
            dtype: torch.dtype | None, data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.g = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.g, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor, input tensor of shape (batch_size, sequence_length, d_model)
        
        Returns:
            torch.Tensor, normalized tensor of the same shape as input
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        result = x / rms * self.g

        return result.to(in_dtype)