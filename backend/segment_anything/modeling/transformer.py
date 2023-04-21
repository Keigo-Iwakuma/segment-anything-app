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


class TwoWayAttetionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross-attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross-attention of dense inputs to sparse inputs.

        Args:
            embedding_dim: The embedding dimension of the inputs.
            num_heads: The number of attention heads.
            mlp_dim: The dimension of the mlp block.
            activation: The activation function to use in the mlp block.
            attention_downsample_rate: The rate at which to downsample the
                embedding dimension of the attention layers.
            skip_first_layer_pe: Whether to skip positional encoding on the
                first layer.
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_head)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe


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
    
    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        """
        Reshape the tensor so that the last dimension becomes the number of heads
        and the second to last dimension becomes the sequence length.
        """
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head
    
    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax

        # Get output
        out = attn @ v  # B x N_heads x N_tokens x C_per_head
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
