import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Args:
            num_embeddings: int, size of the vocabulary
            embedding_dim: int, dimension of the embedding vectors (d_model)
            device: torch.device | None, device to store the parameters on
            dtype: torch.dtype | None, data type of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedding_matrix = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embedding_matrix, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: torch.Tensor, LongTensor of shape (batch_size, sequence_length) containing token IDs
        
        Returns:
            torch.Tensor, embedding vectors of shape (batch_size, sequence_length, embedding_dim)
        """
        return self.embedding_matrix[token_ids]