from functools import partial
from typing import Optional, Literal
import torch
from torch import nn
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

from .embeddings import Timesteps, TimestepEmbedding
from .attention import SpatialAttentionBlock, TemporalAttentionBlock
from .resnet_condition import ResnetBlock, Downsample, Upsample
from .utils import default


class NoiseLevelSequential(nn.Sequential):
    """
    Sequential module that passes the conditioning embedding (noise level + optional
    external conditioning) to each ResnetBlock in the sequence if it accepts it.
    """

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor):
        for module in self:
            if isinstance(module, ResnetBlock):
                x = module(x, cond_emb)
            else:
                x = module(x)
        return x


class Unet3D(nn.Module):
    # Special thanks to lucidrains for the implementation of the base Diffusion model
    # https://github.com/lucidrains/denoising-diffusion-pytorch

    def __init__(
        self,
        dim: int,
        init_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        external_cond_dim: Optional[int] = None,
        channels=3,
        resnet_block_groups=8,
        dim_mults=[1, 2, 4, 8],
        attn_resolutions=[1, 2, 4, 8],
        attn_dim_head=32,
        attn_heads=4,
        use_linear_attn=True,
        use_init_temporal_attn=True,
        init_kernel_size=7,
        is_causal=True,
        time_emb_type: Literal["sinusoidal", "rotary"] = "rotary",
    ):
        super().__init__()

        self.channels = channels
        self.external_cond_dim = external_cond_dim
        self.is_causal = is_causal

        init_dim = default(init_dim, dim)
        out_dim = default(out_dim, channels)
        dim_mults = list(dim_mults)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        mid_dim = dims[-1]

        # ---------------------------------------------------------------------
        # Noise level (timestep) embedding
        # ---------------------------------------------------------------------
        noise_level_emb_dim = dim * 4
        self.noise_level_pos_embedding = nn.Sequential(
            Timesteps(dim, True, 0),
            TimestepEmbedding(in_channels=dim, time_embed_dim=noise_level_emb_dim),
        )

        # ---------------------------------------------------------------------
        # External conditioning embedding (new)
        # Maps external_cond_dim -> noise_level_emb_dim, then concatenated with noise emb
        # ---------------------------------------------------------------------
        if external_cond_dim is not None:
            self.external_cond_mlp = nn.Sequential(
                nn.Linear(external_cond_dim, noise_level_emb_dim),
                nn.GELU(),
                nn.Linear(noise_level_emb_dim, noise_level_emb_dim),
            )
        else:
            self.external_cond_mlp = None

        # Total conditioning dimension passed into ResnetBlock FiLM MLP:
        #   - noise_level_emb                     (always)
        #   + external_cond_emb (if provided)     (optional)
        total_cond_emb_dim = noise_level_emb_dim + (noise_level_emb_dim if external_cond_dim is not None else 0)

        # ---------------------------------------------------------------------
        # Temporal rotary embedding for attention
        # ---------------------------------------------------------------------
        self.rotary_time_pos_embedding = RotaryEmbedding(dim=attn_dim_head) if time_emb_type == "rotary" else None

        # ---------------------------------------------------------------------
        # Initial conv + (optional) temporal attention
        # ---------------------------------------------------------------------
        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(
            channels,
            init_dim,
            kernel_size=(1, init_kernel_size, init_kernel_size),
            padding=(0, init_padding, init_padding),
        )

        self.init_temporal_attn = (
            TemporalAttentionBlock(
                dim=init_dim,
                heads=attn_heads,
                dim_head=attn_dim_head,
                is_causal=is_causal,
                rotary_emb=self.rotary_time_pos_embedding,
            )
            if use_init_temporal_attn
            else nn.Identity()
        )

        # ---------------------------------------------------------------------
        # Down / up path definitions
        # ---------------------------------------------------------------------
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        # ResNet blocks that receive the conditioning embedding (noise + external)
        block_klass_noise = partial(ResnetBlock, groups=resnet_block_groups, emb_dim=total_cond_emb_dim)

        spatial_attn_klass = partial(SpatialAttentionBlock, heads=attn_heads, dim_head=attn_dim_head)
        temporal_attn_klass = partial(
            TemporalAttentionBlock,
            heads=attn_heads,
            dim_head=attn_dim_head,
            is_causal=is_causal,
            rotary_emb=self.rotary_time_pos_embedding,
        )

        curr_resolution = 1

        # --------------------
        # Downsampling blocks
        # --------------------
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx == len(in_out) - 1
            use_attn = curr_resolution in attn_resolutions

            self.down_blocks.append(
                nn.ModuleList(
                    [
                        NoiseLevelSequential(
                            block_klass_noise(dim_in, dim_out),
                            block_klass_noise(dim_out, dim_out),
                            (
                                spatial_attn_klass(
                                    dim_out,
                                    use_linear=use_linear_attn and not is_last,
                                )
                                if use_attn
                                else nn.Identity()
                            ),
                            temporal_attn_klass(dim_out) if use_attn else nn.Identity(),
                        ),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            curr_resolution *= 2 if not is_last else 1

        # -------------
        # Middle block
        # -------------
        self.mid_block = NoiseLevelSequential(
            block_klass_noise(mid_dim, mid_dim),
            spatial_attn_klass(mid_dim),
            temporal_attn_klass(mid_dim),
            block_klass_noise(mid_dim, mid_dim),
        )

        # --------------------
        # Upsampling blocks
        # --------------------
        for idx, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = idx == len(in_out) - 1
            use_attn = curr_resolution in attn_resolutions

            self.up_blocks.append(
                NoiseLevelSequential(
                    block_klass_noise(dim_out * 2, dim_in),
                    block_klass_noise(dim_in, dim_in),
                    (spatial_attn_klass(dim_in, use_linear=use_linear_attn and idx > 0) if use_attn else nn.Identity()),
                    temporal_attn_klass(dim_in) if use_attn else nn.Identity(),
                    Upsample(dim_in) if not is_last else nn.Identity(),
                )
            )

            curr_resolution //= 2 if not is_last else 1

        # Final output head: note this one uses block_klass (no conditioning emb)
        self.out = nn.Sequential(block_klass(dim * 2, dim), nn.Conv3d(dim, out_dim, 1))

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        is_causal: Optional[bool] = None,
    ):
        if is_causal is not None and is_causal != self.is_causal:
            raise ValueError("is_causal must be the same as the one used during initialization")

        # noise_levels originally comes as (frames, batch); rearrange to (batch, frames)
        noise_levels = rearrange(noise_levels, "f b -> b f")

        # ---------------------------------------------------------------------
        # Compute noise level embedding
        # ---------------------------------------------------------------------
        noise_level_emb = self.noise_level_pos_embedding(noise_levels)  # [B, noise_level_emb_dim]

        # ---------------------------------------------------------------------
        # Compute external conditioning embedding (if enabled), and concatenate
        # ---------------------------------------------------------------------
        if self.external_cond_mlp is not None:
            if external_cond is None:
                # If no external_cond is provided at runtime, default to zeros
                external_cond = torch.zeros(
                    noise_level_emb.shape[0],
                    self.external_cond_dim,
                    device=noise_level_emb.device,
                    dtype=noise_level_emb.dtype,
                )

            # ---- external_cond can be MANY shapes; normalize it ----
            if external_cond is not None:

                # Case 1: (T, B, D)  --> reorder to (B, T, D)
                if external_cond.dim() == 3:
                    if external_cond.shape[0] == noise_level_emb.shape[1]:   # T == F
                        external_cond = external_cond.permute(1, 0, 2)        # (B, F, D)
                    else:
                        raise ValueError(f"External cond has shape {external_cond.shape} but expected T == F={noise_level_emb.shape[1]}")

                # Case 2: (B, D) --> broadcast to (B, F, D)
                elif external_cond.dim() == 2:
                    external_cond = external_cond.unsqueeze(1)                        # (B, 1, D)
                    external_cond = external_cond.expand(-1, noise_level_emb.size(1), -1)   # (B, F, D)

                else:
                    raise ValueError(f"Unsupported external_cond shape {external_cond.shape}")

                # Now map through MLP
                external_emb = self.external_cond_mlp(external_cond)   # (B, F, emb_dim)

            else:
                external_emb = torch.zeros_like(noise_level_emb)
                

            # ---- concatenate ----
            cond_emb = torch.cat([noise_level_emb, external_emb], dim=-1)


        else:
            cond_emb = noise_level_emb  # [B, noise_level_emb_dim]

        # ---------------------------------------------------------------------
        # UNet forward pass with conditioning embedding
        # ---------------------------------------------------------------------
        x_in = self.init_conv(x)
        x_in = self.init_temporal_attn(x_in)
        h = x_in.clone()

        hs = []

        # Down path
        for block, downsample in self.down_blocks:
            h = block(h, cond_emb)
            hs.append(h)
            h = downsample(h)

        # Middle
        h = self.mid_block(h, cond_emb)

        # Up path
        for block in self.up_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = block(h, cond_emb)

        # Final output
        h = torch.cat([h, x_in], dim=1)
        return self.out(h)
