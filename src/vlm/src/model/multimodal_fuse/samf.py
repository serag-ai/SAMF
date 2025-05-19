import torch
import torch.nn as nn
import torch.nn.functional as F


class SAMF(nn.Module):
    def __init__(self, d_3d=256, d_2d=768, d_text=3072, d_proj=3072, seq_len=256):
        super().__init__()

        self.seq_len = seq_len  # Target sequence length

        # Projection layers
        self.project_3d = nn.Linear(d_3d, d_proj)  # Project 3D feature
        self.project_2d = nn.Linear(d_2d, d_proj)  # Project 2D slice feature
        # self.project_text = nn.Linear(d_text, d_proj)  # Project text embeddings

        # Attention mechanisms
        self.softmax_slices = nn.Softmax(dim=1)  # Over slices
        self.softmax_text = nn.Softmax(dim=1)  # Over text tokens

        # Final fusion layer
        self.fusion_layer = nn.Linear(
            d_proj * 3, d_proj
        )  # Combine 3D, 2D, and text into d_proj

    def forward(self, feat_3d, feat_2d, feat_text):
        """
        Inputs:
            feat_3d:   [B, 256]         -> Global 3D volume feature
            feat_2d:   [B, S, 768]      -> Per-slice 2D features
            feat_text: [B, 512, 3072]   -> Text embeddings (512 tokens)

        Output:
            fused_representation: [B, 256, 3072] -> Fully fused representation
        """
        B, S, _ = feat_2d.shape  # S is the number of slices
        _, T, _ = feat_text.shape  # T is the number of tokens

        # --- Step 1: Project features to 3072 ---
        feat_3d_proj = self.project_3d(feat_3d).unsqueeze(1)  # [B, 1, 3072]
        feat_2d_proj = self.project_2d(feat_2d)  # [B, S, 3072]
        feat_text_proj = feat_text  # self.project_text(feat_text)  # [B, T, 3072]

        # --- Step 2: Compute Attention Over 2D Slices ---
        attn_scores_slices = torch.bmm(
            feat_2d_proj, feat_3d_proj.transpose(1, 2)
        )  # [B, S, 1]
        attn_weights_slices = self.softmax_slices(
            attn_scores_slices.squeeze(-1)
        )  # [B, S]
        weighted_2d_feat = torch.bmm(
            attn_weights_slices.unsqueeze(1), feat_2d_proj
        )  # [B, 1, 3072]

        # --- Step 3: Compute Attention Over Text Tokens ---
        attn_scores_text = torch.bmm(
            feat_text_proj, feat_3d_proj.transpose(1, 2)
        )  # [B, T, 1]
        attn_weights_text = self.softmax_text(attn_scores_text.squeeze(-1))  # [B, T]
        weighted_text_feat = torch.bmm(
            attn_weights_text.unsqueeze(1), feat_text_proj
        )  # [B, 1, 3072]

        # --- Step 4: Expand Features to Match Sequence Length (256) ---
        expanded_3d = feat_3d_proj.repeat(1, self.seq_len, 1)  # [B, 256, 3072]
        expanded_2d = weighted_2d_feat.repeat(1, self.seq_len, 1)  # [B, 256, 3072]
        expanded_text = weighted_text_feat.repeat(1, self.seq_len, 1)  # [B, 256, 3072]

        # --- Step 5: Concatenate All Features and Apply Fusion ---
        fused_features = torch.cat(
            [expanded_3d, expanded_2d, expanded_text], dim=-1
        )  # [B, 256, 3072*3]
        fused_representation = self.fusion_layer(fused_features)  # [B, 256, 3072]

        return fused_representation
