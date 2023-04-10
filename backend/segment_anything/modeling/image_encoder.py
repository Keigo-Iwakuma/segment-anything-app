import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

from .common import LayerNorm2d, MLPBlock


# This class and its supporting fuctions below lightly adapted from the VitDet backbone available at: # noqa
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderVit(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): input image size
            patch_size (int): patch size
            in_chans (int): number of input image channels
            embed_dim (int): patch embedding dimension
            depth (int): depth of ViT
            num_heads (int): number of attention heads in each ViT block
            mlp_ratio (float): ratio of mlp hidden dim to embedding dim
            out_chans (int): number of output channels
            qkv_bias (bool): enable bias for qkv if True
            norm_layer (nn.Module): normalization layer
            act_layer (nn.Module): activation layer
            use_abs_pos (bool): use absolute position embedding
            use_rel_pos (bool): use relative position embedding
            rel_pos_zero_init (bool): use zero init for relative position bias
            window_size (int): window size for window attention blocks
            global_attn_indexes (tuple[int]): indexes for blocks using global attention
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed


def get_rel_pos(
    q_size: int,
    k_size: int,
    rel_pos: torch.Tensor,
) -> torch.Tensor:
    """
    Get relative position embedding according to the relative positions of query and key size.

    Args:
        q_size (int): query size
        k_size (int): key size
        rel_pos (torch.Tensor): relative position embedding (L, C)

    Returns:
        torch.Tensor: relative position embedding
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], 1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    
    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.)
    k_coords = torch.arange(k_size)[:, None] * max(q_size / k_size, 1.)
    relative_coords = (q_coords + k_coords) + (k_size - 1) * max(q_size / k_size, 1.)

    return rel_pos_resized[relative_coords.long()]


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (tuple[int]): kernel size of the projection layer
            stride (tuple[int]): stride of the projection layer
            padding (tuple[int]): padding of the projection layer
            in_chans (int): number of input image channels
            embed_dim (int): patch embedding dimension
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W => B H W C
        x = x.permute(0, 2, 3, 1)
        return x