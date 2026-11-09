import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        Args:
            in_features: int, final dimension of the input
            out_features: int, final dimension of the output
            device: torch.device | None, device to store the parameters on
            dtype: torch.dtype | None, data type of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        std = math.sqrt(2 / (in_features + out_features))
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor, input tensor of shape (batch_size, in_features)
        
        Returns:
            torch.Tensor, output tensor of shape (batch_size, out_features)
        """
        return torch.einsum('...i,oi->...o', x, self.W)