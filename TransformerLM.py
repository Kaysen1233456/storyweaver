import torch
import torch.nn as nn
from embedding import Embedding
from TransformerBlock import TransformerBlock
from RMSNorm import RMSNorm
from linear import Linear

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, theta: float = 10000.0, dropout: float = 0.0, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Args:
            vocab_size: int, size of the vocabulary
            context_length: int, maximum context length
            d_model: int, dimensionality of the Transformer block inputs
            num_layers: int, number of Transformer blocks
            num_heads: int, number of heads in multi-head self-attention
            d_ff: int, dimensionality of the feed-forward inner layer
            theta: float, RoPE parameter
            dropout: float, dropout on embeddings and each residual branch (0 = off)
            device: torch.device | None, device to store parameters on
            dtype: torch.dtype | None, data type of parameters
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff

        # Token embedding
        # self.token_embeddings = nn.Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.emb_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, theta, dropout=dropout, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])

        # Final layer normalization
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)

        # Language model head
        # self.lm_head = nn.Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            in_indices: torch.Tensor, input indices of shape (batch_size, sequence_length)
        
        Returns:
            torch.Tensor, output tensor of shape (batch_size, sequence_length, vocab_size)
        """
        if in_indices.shape[1] > self.max_seq_len:
            raise ValueError(f"Sequence length {in_indices.shape[1]} exceeds context_length {self.max_seq_len}")
        
        # Embed input indices
        x = self.emb_drop(self.token_embeddings(in_indices))  # Shape: (batch_size, sequence_length, d_model)

        # Apply Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.ln_final(x)

        return self.lm_head(x) # Shape: (batch_size, sequence_length, vocab_size)