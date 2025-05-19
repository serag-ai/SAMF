from torch import nn
import torch
from .samf import SAMF
from .ao2d import AttentionOver2DSlices


class IdentityModule(nn.Module):
    def __init__(self):
        super(IdentityModule, self).__init__()

    def forward(
        self, feat_2d: torch.Tensor, feat_3d: torch.Tensor, feat_text: torch.Tensor
    ):
        return feat_3d  # Returns feat_3d unchanged


def build_mm_fusion(config, delay_load=False, **kwargs):
    fusion_type = getattr(config, "mm_fuse_type")
    if fusion_type == "samf":
        return SAMF()
    elif fusion_type == "ao2d":
        return AttentionOver2DSlices(
            d_3d=256, d_2d=768, d_text=3072, d_proj=3072, seq_len=256
        )
    elif fusion_type == "identity":
        return IdentityModule()
    else:
        raise ValueError(f"Unknown projector type: {fusion_type}")
