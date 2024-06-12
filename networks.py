# This file contains the network architectures and models used in the project's scripts.
# It includes the implementations of the 3D U-Net, our Conditional 3D U-Net,
# the Video Diffusion Models (VDM), and our Video to Video Diffusion Model (Vid2Vid-DM).

import math
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from einops_exts import check_shape, rearrange_many
from rotary_embedding_torch import RotaryEmbedding
from torch import einsum, nn
from tqdm import tqdm


class EMA():
    def __init__(self, beta: float):
        """
        Exponential Moving Average (EMA) class.

        Args:
            beta (float): The decay factor for the moving average.
        """
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model: nn.Module, current_model: nn.Module) -> None:
        """
        Updates the moving average of the model's parameters.

        Args:
            ma_model (nn.Module): The model with the moving average parameters.
            current_model (nn.Module): The model with the current parameters.
        """
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old: Optional[torch.Tensor], new: torch.Tensor) -> torch.Tensor:
        """
        Updates the moving average of a parameter tensor.

        Args:
            old (Optional[torch.Tensor]): The previous moving average tensor.
            new (torch.Tensor): The new tensor to be incorporated into the moving average.

        Returns:
            torch.Tensor: The updated moving average tensor.
        """
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def normalize_img(img: torch.Tensor) -> torch.Tensor:
    """
    Normalize an image tensor to the range [-1, 1].

    Args:
        img (torch.Tensor): Input image tensor.

    Returns:
        torch.Tensor: Normalized image tensor.
    """
    return img * 2 - 1


def unnormalize_img(img: torch.Tensor) -> torch.Tensor:
    """
    Unnormalize an image tensor from the range [-1, 1] to [0, 1].

    Args:
        img (torch.Tensor): Input image tensor.

    Returns:
        torch.Tensor: Unnormalized image tensor.
    """
    return (img + 1) * 0.5


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int]) -> torch.Tensor:
    """
    Extract elements from tensor 'a' using indices 't' and reshape the result.

    Args:
        a (torch.Tensor): Input tensor.
        t (torch.Tensor): Index tensor.
        x_shape (Tuple[int]): Shape of the output tensor (excluding batch dimension).

    Returns:
        torch.Tensor: Extracted and reshaped tensor.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Generate a cosine beta schedule for use in training.

    Args:
        timesteps (int): Number of timesteps in the schedule.
        s (float): Offset parameter for the cosine function.

    Returns:
        torch.Tensor: Cosine beta schedule tensor.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)


class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads: int = 8,
        num_buckets: int = 32,
        max_distance: int = 128
    ):
        """
        Relative Position Bias module.

        Args:
            heads (int): Number of attention heads.
            num_buckets (int): Number of buckets for relative positions.
            max_distance (int): Maximum distance for relative positions.
        """
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        num_buckets: int = 32,
        max_distance: int = 128
    ) -> torch.Tensor:
        """
        Computes the bucket index for a given relative position.

        Args:
            relative_position (torch.Tensor): Relative position tensor.
            num_buckets (int): Number of buckets.
            max_distance (int): Maximum distance.

        Returns:
            torch.Tensor: Bucket index tensor.
        """
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance /
                                                        max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n: int, device: torch.device) -> torch.Tensor:
        """
        Forward pass of the RelativePositionBias module.

        Args:
            n (int): Sequence length.
            device (torch.device): Device to be used.

        Returns:
            torch.Tensor: Relative position bias tensor.
        """
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        """
        Residual module that adds the input tensor to the output of the given function.

        Args:
            fn (nn.Module): The function or module to apply.
        """
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass of the Residual module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor with the residual connection added.
        """
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        """
        Sinusoidal Position Embedding module.

        Args:
            dim (int): Dimension of the position embedding.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SinusoidalPosEmb module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Position embedding tensor.
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim: int) -> nn.Module:
    """
    Upsampling module using 3D transposed convolution.

    Args:
        dim (int): Number of input and output channels.

    Returns:
        nn.Module: Upsampling module.
    """
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


def Downsample(dim: int) -> nn.Module:
    """
    Downsampling module using 3D convolution.

    Args:
        dim (int): Number of input and output channels.

    Returns:
        nn.Module: Downsampling module.
    """
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Layer normalization module.

        Args:
            dim (int): Number of input channels.
            eps (float): Small value added to the denominator for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LayerNorm module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized output tensor.
        """
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        """
        PreNormalization module that applies layer normalization before the given function.

        Args:
            dim (int): Number of input channels.
            fn (nn.Module): The function or module to apply.
        """
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the PreNorm module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying layer normalization and the function.
        """
        x = self.norm(x)
        return self.fn(x, **kwargs)


class Block(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        """
        Block module consisting of convolution, group normalization, and activation.

        Args:
            dim (int): Number of input channels.
            dim_out (int): Number of output channels.
            groups (int): Number of groups for group normalization.
        """
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the Block module.

        Args:
            x (torch.Tensor): Input tensor.
            scale_shift (torch.Tensor): Scale and shift parameters for conditional computation.

        Returns:
            torch.Tensor: Output tensor after applying convolution, normalization, and activation.
        """
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, time_emb_dim: int = None, groups: int = 8):
        """
        Residual block module with optional time embedding.

        Args:
            dim (int): Number of input channels.
            dim_out (int): Number of output channels.
            time_emb_dim (int): Dimension of the time embedding. Defaults to None.
            groups (int): Number of groups for group normalization. Defaults to 8.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim is not None else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the ResnetBlock module.

        Args:
            x (torch.Tensor): Input tensor.
            time_emb (torch.Tensor): Time embedding tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying the residual block.
        """
        scale_shift = None
        if self.mlp is not None:
            assert time_emb is not None, 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        """
        Spatial Linear Attention module.

        Args:
            dim (int): Number of input channels.
            heads (int): Number of attention heads. Defaults to 4.
            dim_head (int): Dimension of each attention head. Defaults to 32.
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SpatialLinearAttention module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying spatial linear attention.
        """
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(
            qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops: str, to_einops: str, fn):
        """
        Module that converts input tensor from one einops format to another format using a given function.

        Args:
            from_einops (str): Source einops format.
            to_einops (str): Target einops format.
            fn: Function to be applied to the tensor in the target format.
        """
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the EinopsToAndFrom module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after converting to the target einops format and applying the function.
        """
        shape = x.shape
        reconstitute_kwargs = dict(
            tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(
            x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        rotary_emb=None
    ):
        """
        Attention module that performs multi-head self-attention on the input tensor.

        Args:
            dim (int): Input tensor dimension.
            heads (int): Number of attention heads.
            dim_head (int): Dimension of each attention head.
            rotary_emb: Rotary positional embedding module.
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        pos_bias: torch.Tensor = None,
        focus_present_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): Input tensor.
            pos_bias (torch.Tensor): Relative positional bias tensor.
            focus_present_mask (torch.Tensor): Mask indicating which tokens are focusing on the present.

        Returns:
            torch.Tensor: Output tensor after performing self-attention.
        """
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if focus_present_mask is not None and focus_present_mask.all():
            values = qkv[-1]
            return self.to_out(values)

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)

        q = q * self.scale

        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        if pos_bias is not None:
            sim = sim + pos_bias

        if focus_present_mask is not None and not (~focus_present_mask).all():
            attend_all_mask = torch.ones(
                (n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)


# Cond_UNet model
class Cond_UNet(nn.Module):
    def __init__(
        self,
        im_size: int,
        dim_mults: tuple = (1, 2, 4, 8),
        channels: int = 3,
        cond_channels: int = 3,
        attn_heads: int = 8,
        attn_dim_head: int = 32,
        init_kernel_size: int = 7,
        resnet_groups: int = 8
    ):
        """
        Conditional UNet model, used as the Vdieo UNet of the Vid2Vid-DM.

        Args:
            im_size (int): Size of the input frame (assumed to be square).
            dim_mults (tuple): Dimension multipliers for different levels of the UNet.
            channels (int): Number of input channels.
            cond_channels (int): Number of conditional channels (e.g., for conditioning on depth video).
            attn_heads (int): Number of attention heads.
            attn_dim_head (int): Dimension of each attention head.
            init_kernel_size (int): Kernel size for the initial convolutional layer.
            resnet_groups (int): Number of groups for the ResNet blocks.
        """
        super().__init__()
        self.im_size = im_size
        self.dim_mults = dim_mults
        self.channels = channels
        self.cond_channels = cond_channels
        self.in_channels = self.channels + self.cond_channels
        self.attn_heads = attn_heads
        self.attn_dim_head = attn_dim_head
        self.init_kernel_size = init_kernel_size
        self.resnet_groups = resnet_groups

        # temporal attention and its relative positional encoding
        rotary_emb = RotaryEmbedding(min(32, self.attn_dim_head))

        def temporal_attn(dim: int) -> EinopsToAndFrom:
            """
            Create a temporal attention module.

            Args:
                dim (int): Input dimension.

            Returns:
                EinopsToAndFrom: Temporal attention module wrapped in an EinopsToAndFrom module.
            """
            return EinopsToAndFrom(
                'b c f h w', 'b (h w) f c',
                Attention(dim, heads=self.attn_heads, dim_head=self.attn_dim_head,
                          rotary_emb=rotary_emb))

        self.time_rel_pos_bias = RelativePositionBias(
            heads=self.attn_heads, max_distance=32)

        # initial conv
        assert (self.init_kernel_size % 2) == 1

        self.init_padding = self.init_kernel_size // 2
        self.init_conv = nn.Conv3d(self.in_channels, self.im_size, (
            1, self.init_kernel_size, self.init_kernel_size), padding=(0, self.init_padding, self.init_padding))

        self.init_temporal_attn = Residual(
            PreNorm(self.im_size, temporal_attn(self.im_size)))

        # dimensions
        self.dims = [self.im_size, *
                     map(lambda m: self.im_size * m, self.dim_mults)]
        self.in_out = list(zip(self.dims[:-1], self.dims[1:]))

        # time conditioning
        self.time_dim = self.im_size * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.im_size),
            nn.Linear(self.im_size, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        self.num_resolutions = len(self.in_out)

        # block type
        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=self.time_dim)

        # modules for all downstream path layers
        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (self.num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in*2, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(
                    dim_out, heads=self.attn_heads))),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        self.mid_dim = self.dims[-1]
        self.mid_block1 = block_klass_cond(self.mid_dim, self.mid_dim)

        self.spatial_attn = EinopsToAndFrom(
            'b c f h w', 'b f (h w) c', Attention(self.mid_dim, heads=self.attn_heads))

        self.mid_spatial_attn = Residual(
            PreNorm(self.mid_dim, self.spatial_attn))
        self.mid_temporal_attn = Residual(
            PreNorm(self.mid_dim, temporal_attn(self.mid_dim)))

        self.mid_block2 = block_klass_cond(self.mid_dim, self.mid_dim)

        # modules for all upstream path layers
        for ind, (dim_in, dim_out) in enumerate(reversed(self.in_out)):
            is_last = ind >= (self.num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(
                    dim_in, heads=self.attn_heads))),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            block_klass(self.im_size * 2, self.im_size),
            nn.Conv3d(self.im_size, self.channels, 1)
        )

    def forward(
            self,
            x: torch.Tensor,
            time: torch.Tensor,
            cond_list: List[torch.Tensor] = []) -> torch.Tensor:
        """
        Forward pass of the Cond_UNet model.

        Args:
            x (torch.Tensor): Input video tensor.
            time (torch.Tensor): Time tensor.
            cond_list (List[torch.Tensor]): List of output tensors from the UNet model.

        Returns:
            torch.Tensor: Output tensor of the model.
        """
        batch, device = x.shape[0], x.device

        focus_present_mask = torch.zeros(
            (batch,), device=device, dtype=torch.bool)

        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)

        x = torch.cat((x, cond_list.pop()), dim=1)

        x = self.init_conv(x)

        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)

        r = x.clone()

        t = self.time_mlp(time) if self.time_mlp is not None else None

        h = []

        # Downstream
        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = torch.cat((x, cond_list.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(
            x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, t)

        # Upstream
        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)

# UNet model


class UNet(nn.Module):
    def __init__(
        self,
        im_size: int,
        out_list: bool = False,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        channels: int = 3,
        attn_heads: int = 8,
        attn_dim_head: int = 32,
        init_kernel_size: int = 7,
        resnet_groups: int = 8
    ):
        """
        Initializes the UNet model.

        Args:
            im_size (int): Input frame size.
            out_list (bool): Whether to output a list of intermediate feature maps (True used for Vid2Vid-DM).
            dim_mults (Tuple[int]): Dimension multipliers for each U-Net layer.
            channels (int): Number of input channels.
            attn_heads (int): Number of attention heads.
            attn_dim_head (int): Dimension of each attention head.
            init_kernel_size (int): Kernel size for the initial convolution.
            resnet_groups (int): Number of groups for GroupNorm in the residual blocks.
        """
        super().__init__()
        self.im_size = im_size
        self.out_list = out_list
        self.dim_mults = dim_mults
        self.channels = channels
        self.attn_heads = attn_heads
        self.attn_dim_head = attn_dim_head
        self.init_kernel_size = init_kernel_size
        self.resnet_groups = resnet_groups

        # temporal attention and its relative positional encoding
        rotary_emb = RotaryEmbedding(min(32, self.attn_dim_head))

        def temporal_attn(dim): return EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(
            dim, heads=self.attn_heads, dim_head=self.attn_dim_head, rotary_emb=rotary_emb))

        # realistically will not be able to generate that many frames of video... yet
        self.time_rel_pos_bias = RelativePositionBias(
            heads=self.attn_heads, max_distance=32)

        # initial conv
        assert (self.init_kernel_size % 2) == 1

        self.init_padding = self.init_kernel_size // 2
        self.init_conv = nn.Conv3d(self.channels, self.im_size, (1, self.init_kernel_size,
                                   self.init_kernel_size), padding=(0, self.init_padding, self.init_padding))

        self.init_temporal_attn = Residual(
            PreNorm(self.im_size, temporal_attn(self.im_size)))

        # dimensions
        self.dims = [self.im_size, *map(lambda m: self.im_size * m, dim_mults)]
        self.in_out = list(zip(self.dims[:-1], self.dims[1:]))

        # time conditioning
        self.time_dim = self.im_size * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.im_size),
            nn.Linear(self.im_size, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        self.num_resolutions = len(self.in_out)

        # block type
        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=self.time_dim)

        # modules for all downstream path layers
        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (self.num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(
                    dim_out, heads=self.attn_heads))),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        self.mid_dim = self.dims[-1]
        self.mid_block1 = block_klass_cond(self.mid_dim, self.mid_dim)

        self.spatial_attn = EinopsToAndFrom(
            'b c f h w', 'b f (h w) c', Attention(self.mid_dim, heads=self.attn_heads))

        self.mid_spatial_attn = Residual(
            PreNorm(self.mid_dim, self.spatial_attn))
        self.mid_temporal_attn = Residual(
            PreNorm(self.mid_dim, temporal_attn(self.mid_dim)))

        self.mid_block2 = block_klass_cond(self.mid_dim, self.mid_dim)

        # modules for all upstream path layers
        for ind, (dim_in, dim_out) in enumerate(reversed(self.in_out)):
            is_last = ind >= (self.num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(
                    dim_in, heads=attn_heads))),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.out_dims = [self.channels] + self.dims[:self.num_resolutions]
        self.out_res = [int(self.im_size*self.im_size/list_dim)
                        for list_dim in self.dims]
        self.final_conv = nn.Sequential(
            block_klass(self.im_size * 2, self.im_size),
            nn.Conv3d(self.im_size, self.channels, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        cond_dropout: float = 0.
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the UNet model.

        Args:
            x (torch.Tensor): Input video tensor.
            time (torch.Tensor): Time tensor.
            cond_dropout (float): Dropout probability for the conditional. Defaults to 0.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: Output tensor or a list of intermediate feature maps 
            depending on the 'out_list' argument passed during initialization.
        """
        batch, frames, device = x.shape[0], x.shape[2], x.device

        if torch.rand(1) <= cond_dropout:
            output = []
            for dim, res in zip(reversed(self.out_dims), reversed(self.out_res)):
                output.append(torch.zeros(
                    (batch, dim, frames, res, res), device=device))
            return output

        focus_present_mask = torch.zeros(
            (batch,), device=device, dtype=torch.bool)

        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)

        x = self.init_conv(x)

        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)

        r = x.clone()

        t = self.time_mlp(time) if self.time_mlp is not None else None

        h = []

        # Downstream
        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(
            x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, t)

        output = []

        # Upstream
        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias=time_rel_pos_bias,
                              focus_present_mask=focus_present_mask)
            output.append(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_conv(x)
        output.append(x)

        if self.out_list:
            return output
        else:
            return x

# VDM gaussian diffusion model


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn: nn.Module,
        image_size: int,
        num_frames: int,
        channels: int = 3,
        timesteps: int = 1000
    ):
        """
        Gaussian Diffusion model for video denoising.

        Args:
            denoise_fn (nn.Module): Denoising model.
            image_size (int): Size of the input frames (assumed to be square).
            num_frames (int): Number of frames of the videos.
            channels (int): Number of frames channels. Defaults to 3.
            timesteps (int): Number of time steps in the diffusion process. Defaults to 1000.
        """
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # register buffer helper function that casts float64 to float32
        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas *
                        torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev)
                        * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def q_mean_variance(self, x_start: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the mean, variance, and log of variance of the q distribution.

        Args:
            x_start (torch.Tensor): Input tensor at the start.
            t (int): Time step.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Mean, variance, and log of variance tensors.
        """
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t: torch.Tensor, t: int, noise: torch.Tensor) -> torch.Tensor:
        """
        Predict the start tensor from the noise tensor.

        Args:
            x_t (torch.Tensor): Input tensor at time t.
            t (int): Time step.
            noise (torch.Tensor): Noise tensor.

        Returns:
            torch.Tensor: Predicted start tensor.
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the posterior mean, variance, and clipped log of variance of the q distribution.

        Args:
            x_start (torch.Tensor): Start tensor.
            x_t (torch.Tensor): Input tensor at time t.
            t (int): Time step.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Posterior mean, variance, and clipped logarithm of variance tensors.
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the mean, variance, and log of variance of the p distribution.

        Args:
            x (torch.Tensor): Input tensor.
            t (int): Time step.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Mean, variance, and log of variance tensors.
        """
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.denoise_fn(x, t))

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """
        Generate a sample from the p distribution.

        Args:
            x (torch.Tensor): Input tensor.
            t (int): Time step.

        Returns:
            torch.Tensor: Sampled tensor.
        """
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t)
        noise = torch.randn_like(x)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Generate a sample by looping over time steps.

        Args:
            shape (Tuple[int, ...]): Shape of the sample.

        Returns:
            torch.Tensor: Sampled tensor.
        """
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long))
        return unnormalize_img(img)

    @torch.inference_mode()
    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Generate samples from the diffusion model.

        Args:
            batch_size (int): Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples.
        """
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        return self.p_sample_loop((batch_size, channels, num_frames, image_size, image_size))

    @torch.inference_mode()
    def noise_sample(self,
                     img: torch.Tensor,
                     noise_timesteps: int,
                     denoise_timestep: int,
                     noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a denoised sample with noise.

        Args:
            img (torch.Tensor): Input video.
            noise_timesteps (int): Number of noise timesteps.
            denoise_timestep (int): Number of denoise timesteps.
            noise (Optional[torch.Tensor]): Noise tensor. If None, random noise will be generated.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the denoised image and the noisy image.
        """
        b, device, img_size, = img.shape[0], img.device, self.image_size
        t = torch.randint(0, noise_timesteps, (b,), device=device).long()
        img = normalize_img(img)

        if noise is None:
            noise = torch.randn_like(img)
        noisy = self.q_sample(x_start=img, t=t, noise=noise)
        recon = noisy
        for i in tqdm(reversed(range(0, denoise_timestep)), desc='denoise sampling loop time step', total=denoise_timestep):
            recon = self.p_sample(recon, torch.full(
                (b,), i, device=device, dtype=torch.long))
        recon = unnormalize_img(recon)

        return recon, noisy

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample from the q distribution.

        Args:
            x_start (torch.Tensor): Starting video.
            t (torch.Tensor): Time step.
            noise (Optional[torch.Tensor]): Noise tensor. If None, random noise will be generated.

        Returns:
            torch.Tensor: Sampled video.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Compute the loss for training the diffusion model.

        Args:
            x_start (torch.Tensor): Starting video.
            t (torch.Tensor): Time step.
            noise (Optional[torch.Tensor]): Noise tensor. If None, random noise will be generated.

        Returns:
            torch.Tensor: Loss value.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.denoise_fn(x_noisy, t, **kwargs)

        loss = F.mse_loss(noise, x_recon)
        return loss

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forard pass of diffusion model.

        Args:
            x (torch.Tensor): Input video tensor.

        Returns:
            torch.Tensor: Loss value.
        """
        b, device, img_size, = x.shape[0], x.device, self.image_size
        check_shape(x, 'b c f h w', c=self.channels,
                    f=self.num_frames, h=img_size, w=img_size)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        x = normalize_img(x)
        return self.p_losses(x, t, *args, **kwargs)

# GD-VDM gaussian diffusion model


class Extented_GaussianDiffusionModel(nn.Module):
    def __init__(
        self,
        denoise_fn: nn.Module,
        cond_fn: nn.Module,
        image_size: int,
        num_frames: int,
        channels: int = 3,
        cond_channels: int = 3,
        timesteps: int = 1000,
        cond_dropout: float = 0.2
    ):
        """
        Video to Video Gaussian Diffusion Model.

        Args:
            denoise_fn (nn.Module): Denoising function/module.
            cond_fn (nn.Module): Conditioning function/module.
            image_size (int): Size of the input frames (assuming square images).
            num_frames (int): Number of frames of the videos.
            channels (int): Number of channels in the input frames. Default is 3.
            cond_channels (int): Number of channels in the conditioning frames. Default is 3.
            timesteps (int): Number of diffusion timesteps. Default is 1000.
            cond_dropout (float): Dropout probability for the conditioning video. Default is 0.2.


        This class represents the Vid2Vid-DM model, which is designed to learn the 
        mapping from one set of video domain(e.g. depth maps videos) to another video doamin (real-world videos).        
        """
        super().__init__()
        self.channels = channels
        self.cond_channels = cond_channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn
        self.cond_fn = cond_fn
        self.cond_dropout = cond_dropout

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # register buffer helper function that casts float64 to float32
        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas *
                        torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev)
                        * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def q_mean_variance(self, x_start: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the mean, variance, and log of variance of the q distribution.

        Args:
            x_start (torch.Tensor): Input tensor at the start.
            t (int): Time step.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Mean, variance, and log of variance tensors.
        """
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t: torch.Tensor, t: int, noise: torch.Tensor) -> torch.Tensor:
        """
        Predict the start tensor from the noise tensor.

        Args:
            x_t (torch.Tensor): Input tensor at time t.
            t (int): Time step.
            noise (torch.Tensor): Noise tensor.

        Returns:
            torch.Tensor: Predicted start tensor.
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the posterior mean, variance, and clipped log of variance of the q distribution.

        Args:
            x_start (torch.Tensor): Start tensor.
            x_t (torch.Tensor): Input tensor at time t.
            t (int): Time step.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Posterior mean, variance, and clipped logarithm of variance tensors.
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x: torch.Tensor, t: int, cond_img: torch.Tensor, cond_t: int, cond_scale: float = 0.):
        """
        Compute the mean, variance, and log variance of the p distribution.

        Args:
            x (torch.Tensor): Input tensor.
            t (int): Current timestep.
            cond_img (torch.Tensor): Conditional video.
            cond_t (int): Timestep of the conditional video.
            cond_scale (float): Scale factor for the conditional information.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing the mean, variance, and log variance.
        """

        if cond_scale == 0:
            cond_list = self.cond_fn.forward(
                x=cond_img, time=cond_t, cond_dropout=0.)
            noise = self.denoise_fn.forward(x=x, time=t, cond_list=cond_list)
        else:
            null_cond_list = self.cond_fn.forward(
                x=cond_img, time=cond_t, cond_dropout=1.)
            null_logits = self.denoise_fn.forward(
                x=x, time=t, cond_list=null_cond_list)

            cond_list = self.cond_fn.forward(
                x=cond_img, time=cond_t, cond_dropout=0.)
            logits = self.denoise_fn.forward(x=x, time=t, cond_list=cond_list)

            noise = logits + cond_scale * (logits - null_logits)

        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, cond_img: torch.Tensor, cond_t: torch.Tensor, cond_scale: float = 0.) -> torch.Tensor:
        """
        Sample from the diffusion process.

        Args:
            x (torch.Tensor): Input tensor.
            t (torch.Tensor): Current timestep.
            cond_img (torch.Tensor): Conditional video.
            cond_t (torch.Tensor): Timestep of the conditional video.
            cond_scale (float): Scale factor for the conditional information.

        Returns:
            torch.Tensor: Sampled tensor.

        """
        b, *_ = x.shape

        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, cond_img=cond_img, cond_t=cond_t, cond_scale=cond_scale)
        noise = torch.randn_like(x)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape: Tuple[int, ...], cond_img: torch.Tensor, cond_scale: float = 0.) -> torch.Tensor:
        """
        Perform sampling loop over timesteps.

        Args:
            shape (Tuple[int, ...]): Shape of the tensor to be sampled.
            cond_img (torch.Tensor): Conditional video.
            cond_scale (float): Scale factor for the conditional information.

        Returns:
            torch.Tensor: Sampled tensor.

        """
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), cond_img, cond_t=torch.full(
                (b,), i, device=device, dtype=torch.long), cond_scale=cond_scale)

        return unnormalize_img(img)

    @torch.inference_mode()
    def sample(self, cond_img: torch.Tensor, cond_scale: float = 0.) -> torch.Tensor:
        """
        Generate a sample.

        Args:
            cond_img (torch.Tensor): Conditional video.
            cond_scale (float): Scale factor for the conditional information.

        Returns:
            torch.Tensor: Generated sample.

        """
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        batch_size, *_ = cond_img.shape

        return self.p_sample_loop((batch_size, channels, num_frames, image_size, image_size), cond_img, cond_scale=cond_scale)

    @torch.inference_mode()
    def noise_sample(self,
                     img: torch.Tensor,
                     noise_timesteps: int,
                     denoise_timestep: int,
                     noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a denoised sample with noise.

        Args:
            img (torch.Tensor): Input video.
            noise_timesteps (int): Number of noise timesteps.
            denoise_timestep (int): Number of denoise timesteps.
            noise (Optional[torch.Tensor]): Noise tensor. If None, random noise will be generated.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the denoised image and the noisy image.
        """
        b, device, img_size, = img.shape[0], img.device, self.image_size
        t = torch.randint(0, noise_timesteps, (b,), device=device).long()
        img = normalize_img(img)

        if noise is None:
            noise = torch.randn_like(img)

        noisy = self.q_sample(x_start=img, t=t, noise=noise)
        recon = noisy
        for i in tqdm(reversed(range(0, denoise_timestep)), desc='denoise sampling loop time step', total=denoise_timestep):
            recon = self.p_sample(recon, torch.full(
                (b,), i, device=device, dtype=torch.long))
        recon = unnormalize_img(recon)

        return recon, noisy

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample from the q distribution.

        Args:
            x_start (torch.Tensor): Starting video.
            t (torch.Tensor): Time step.
            noise (Optional[torch.Tensor]): Noise tensor. If None, random noise will be generated.

        Returns:
            torch.Tensor: Sampled video.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    def p_losses(self,
                 x_start: torch.Tensor,
                 t: torch.Tensor,
                 cond_img: torch.Tensor,
                 cond_t: torch.Tensor,
                 cond_dropout: float = 0.,
                 noise: Optional[torch.Tensor] = None,
                 **kwargs) -> torch.Tensor:
        """
        Compute the loss for training the diffusion model.

        Args:
            x_start (torch.Tensor): Starting video.
            t (torch.Tensor): Time step.
            cond_img (torch.Tensor): Conditional video.
            cond_t (torch.Tensor): Timestep of the conditional video.
            cond_dropout (float): Dropout probability for the conditioning video. Default is 0..
            noise (Optional[torch.Tensor]): Noise tensor. If None, random noise will be generated.

        Returns:
            torch.Tensor: Loss value.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        cond_list = self.cond_fn(
            cond_img, cond_t, cond_dropout=cond_dropout, **kwargs)

        x_recon = self.denoise_fn(x_noisy, t, cond_list=cond_list, **kwargs)

        loss = F.mse_loss(noise, x_recon)

        return loss

    def forward(self,
                x: torch.Tensor,
                cond_img: torch.Tensor,
                cond_dropout: float = None,
                *args,
                **kwargs) -> torch.Tensor:
        """
        Forard pass of diffusion model.

        Args:
            x (torch.Tensor): Input video tensor.
            cond_img (torch.Tensor): Conditional video.
            cond_dropout (float): Dropout probability for the conditioning video. Default is 0..

        Returns:
            torch.Tensor: Loss value.
        """
        b, device, img_size, = x.shape[0], x.device, self.image_size
        check_shape(x, 'b c f h w', c=self.channels,
                    f=self.num_frames, h=img_size, w=img_size)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        cond_t = torch.randint(0, self.num_timesteps,
                               (b,), device=device).long()
        x = normalize_img(x)
        cond_img = normalize_img(cond_img)
        if cond_dropout is None:
            cond_dropout = self.cond_dropout
        return self.p_losses(x, t, cond_img, cond_t, cond_dropout=cond_dropout, *args, **kwargs)
