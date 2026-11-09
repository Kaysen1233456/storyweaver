import torch
import torch.nn as nn
import math

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Args:
            d_model: int, dimensionality of the feedforward input and output
            device: torch.device | None, device to store the parameters on
            dtype: torch.dtype | None, data type of the parameters
        """
        super().__init__()
        self.d_model = d_model

        d_ff_base = int((8 / 3) * d_model)
        self.d_ff = ((d_ff_base + 63) // 64) * 64

        std_w = math.sqrt(2.0 / (self.d_ff + d_model))

        self.w1 = nn.Linear(d_model, self.d_ff, bias=False, device=device, dtype=dtype)
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=std_w, a=-3*std_w, b=3*std_w)
        
        self.w2 = nn.Linear(self.d_ff, d_model, bias=False, device=device, dtype=dtype)
        nn.init.trunc_normal_(self.w2.weight, mean=0.0, std=std_w, a=-3*std_w, b=3*std_w)

        self.w3 = nn.Linear(d_model, self.d_ff, bias=False, device=device, dtype=dtype)
        nn.init.trunc_normal_(self.w3.weight, mean=0.0, std=std_w, a=-3*std_w, b=3*std_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor, input tensor of shape (..., d_model)
        
        Returns:
            torch.Tensor, output tensor of shape (..., d_model)
        """
        w1x = self.w1(x)
        silu = w1x * torch.sigmoid(w1x)
        return self.w2(silu * self.w3(x))