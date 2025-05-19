import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedSumGatingFuser(nn.Module):
    def __init__(self, d_2d, d_output, d_text):
        """
        Initialize the weighted sum gating fusion module.

        Args:
            r (int): Input dimension of 2D slice features
            e (int): Target embedding dimension
            F_dim (int): Input dimension of text features
        """
        super(WeightedSumGatingFuser, self).__init__()
        self.linear_proj_2D = nn.Linear(
            d_2d, d_output
        )  # Project 2D features from r to e
        self.linear_proj_text = nn.Linear(
            d_text, d_output
        )  # Project text features from F_dim to e

    def forward(self, feat_2d, feat_3d, feat_text):
        """
        Perform text-guided weighted sum fusion with gating.

        Args:
            feat_2d (torch.Tensor): 2D slice features of shape (B, S, r)
            feat_3d (torch.Tensor): 3D feature of shape (B, 1, e)
            feat_text (torch.Tensor): Text features of shape (B, F_dim, z)

        Returns:
            torch.Tensor: Fused feature tensor of shape (B, e, z)
        """
        B, S, d_2d = feat_2d.shape
        _, d_text, z = feat_text.shape

        # Project 2D features from r to e
        F_2D_proj = self.linear_proj_2D(feat_2d)  # (B, S, e)

        # Project text features from F_dim to e
        F_text_proj = self.linear_proj_text(feat_text.permute(0, 2, 1))  # (B, z, e)
        F_text_proj = F_text_proj.permute(0, 2, 1)  # (B, e, z)

        # Compute gating scores using dot product
        F_text_exp = F_text_proj.unsqueeze(1)  # (B, 1, e, z)
        F_2D_exp = F_2D_proj.unsqueeze(-1)  # (B, S, e, 1)

        gating_scores = torch.sum(F_2D_exp * F_text_exp, dim=2)  # (B, S, z)

        # Normalize gating scores using softmax along slice dimension (S)
        gating_weights = torch.softmax(gating_scores, dim=1)  # (B, S, z)

        # Compute weighted sum of slices
        F_weighted_sum = torch.sum(
            F_2D_proj.unsqueeze(-1) * gating_weights.unsqueeze(2), dim=1
        )  # (B, e, z)

        # Incorporate 3D feature (broadcasted to match shape)
        F_3D_exp = feat_3d.squeeze(1).unsqueeze(-1)  # (B, e, 1)
        F_fused = F_weighted_sum + F_3D_exp  # (B, e, z)

        return F_fused
