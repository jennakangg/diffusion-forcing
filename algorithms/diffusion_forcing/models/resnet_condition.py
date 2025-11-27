from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        emb_dim: Optional[int] = None,
        groups: int = 8,
        eps: float = 1e-6,
    ):
        """
        3D ResNet block with FiLM conditioning.

        Args:
            dim:       input channels
            dim_out:   output channels
            emb_dim:   conditioning embedding dim (noise + external cond)
            groups:    GroupNorm groups
            eps:       GroupNorm epsilon
        """
        super().__init__()

        # ----- 1st half: norm → activation → conv -----
        self.in_layers = nn.Sequential(
            nn.GroupNorm(groups, dim, eps=eps),
            nn.SiLU(),
            nn.Conv3d(dim, dim_out, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
        )

        # ----- 2nd half: norm → activation → conv -----
        # Split norm out so FiLM scale/shift can apply BEFORE activation
        self.out_norm = nn.GroupNorm(groups, dim_out, eps=eps)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Conv3d(dim_out, dim_out, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
        )

        # ----- FiLM conditioning -----
        # emb_dim → (scale | shift) for each output channel
        if emb_dim is not None:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_dim, dim_out * 2),   # produces [scale | shift]
            )
        else:
            self.emb_layers = None

        # ----- skip connection projection -----
        self.skip_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None):
        """
        x:   [B, C, T, H, W]
        emb: [B, emb_dim]  (global conditioning)
        """

        if emb is not None:
            # Convert (B, F, C) → (B, C)
            if emb.dim() == 3:
                emb = emb.mean(dim=1)

        # First half
        h = self.in_layers(x)

        # Conditioning
        if self.emb_layers is not None:
            assert emb is not None, "This ResnetBlock requires 'emb' but got None."

            # Linear projection → produce scale+shift
            emb_out = self.emb_layers(emb)          # [B, 2*dim_out]

            # Broadcast over time & space
            emb_out = rearrange(emb_out, "b c -> b c 1 1 1")

            scale, shift = emb_out.chunk(2, dim=1)  # both [B, dim_out, 1, 1, 1]

            # FiLM: normalize then apply scale/shift
            h = self.out_norm(h) * (1 + scale) + shift
            h = self.out_layers(h)

        else:
            # No conditioning
            h = self.out_norm(h)
            h = self.out_layers(h)

        # Residual connection
        return self.skip_conv(x) + h
class Downsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv3d(
            dim, dim, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def forward(self, x):
        x = F.interpolate(x, scale_factor=[1.0, 2.0, 2.0], mode="nearest")
        return self.conv(x)