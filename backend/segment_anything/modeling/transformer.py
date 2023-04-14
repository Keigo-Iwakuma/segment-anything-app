import torch
from torch import Tensor, nn

import math
from typing import Tuple, Type

from .common import MLPBlock


class TwoWayTransformer(nn.Module):
    def __init__(
        self
    ) -> None:
        pass


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self, 
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % self.num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
    
    def _separate_heads(self, x: Tensor) -> Tensor:
        """
        Reshape the tensor so that the last dimension becomes the number of heads
        and the second to last dimension becomes the sequence length.
        """
        b, n, c = x.shape
        x = x.reshape()