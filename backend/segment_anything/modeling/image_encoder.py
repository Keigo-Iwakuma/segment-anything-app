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


class Attention(nn.Module):
    """Multi-head attention with relative position encoding."""    

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): number of input channels
            num_heads (int): number of attention heads
            qkv_bias (bool): enable bias for qkv if True
            use_rel_pos (bool): use relative position embedding
            rel_pos_zero_init (bool): use zero init for relative position bias
            input_size (tuple[int]): height and width of input
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "input_size must be specified when use_rel_pos is True"
            # initialize relative position embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros((2 * input_size[0] - 1, head_dim)))
            self.rel_pos_w = nn.Parameter(torch.zeros((2 * input_size[1] - 1, head_dim)))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B, nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)) 
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed
    Args:
        x (torch.Tensor): input tokens with shape (B, C, H, W)
        window_size (int): window size
    
    Returns:
        windows: windows after partition with shape (num_windows*B, C, window_size, window_size)
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_h, 0, pad_w))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequence and removing padding
    Args:
        windows (torch.Tensor): windows after partition with shape (num_windows*B, window_size, window_size, C)
        window_size (int): window size
        pad_hw (tuple[int]): height and width of padding (Hp, Wp)
        hw (tuple[int]): height and width of original sequence (H, W)
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp // window_size * Wp // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


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


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed relative positional embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950

    Args:
        attn (torch.Tensor): attention map
        q (torch.Tensor): query q in the attention layer with shape (B, q_h * q_w, C)
        rel_pos_h (torch.Tensor): relative position embedding (Lh, C) for height axis
        rel_pos_w (torch.Tensor): relative position embedding (Lw, C) for width axis
        q_size (tuple[int]): spatial sequence size of query q with (q_h, q_w)
        k_size (tuple[int]): spatial sequence size of key k with (k_h, k_w)
    
    Returns:
        attn (torch.Tensor): attention map with added relative positional embeddings
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


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