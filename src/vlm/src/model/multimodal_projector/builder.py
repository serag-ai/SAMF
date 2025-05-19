from torch import nn
import torch
from .spatial_pooling_projector import SpatialPoolingProjector


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class Minigpt(nn.Module):
    def __init__(self, config=None):
        super(Minigpt, self).__init__()
        # c*4 is the input size, and c is the output size for the linear layer
        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.linear = nn.Linear(inc * 4, ouc)

    def forward(self, x):
        # x is the input tensor with shape [b, num_tokens, c]
        b, num_tokens, c = x.shape

        # Check if num_tokens is divisible by 4
        if num_tokens % 4 != 0:
            raise ValueError("num_tokens must be divisible by 4")

        # Reshape x to [b, num_tokens/4, c*4]
        x = x.view(b, num_tokens // 4, c * 4)

        # Apply the linear transformation
        x = self.linear(x)
        return x


class Vanilla(nn.Module):
    def __init__(self, config=None):
        super(Vanilla, self).__init__()
        # c*4 is the input size, and c is the output size for the linear layer
        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.linear = nn.Linear(inc * 4, ouc)

    def forward(self, x):
        b, num_tokens, c = x.shape

        # Check if num_tokens is divisible by 4
        if num_tokens % 4 != 0:
            raise ValueError("num_tokens must be divisible by 4")

        # First, reshape to [b, num_tokens//4, 4, c]
        x = x.view(b, num_tokens // 4, 4, c)

        # Then, permute to interleave the tokens
        x = x.permute(0, 1, 3, 2).contiguous()

        # Finally, reshape to [b, num_tokens//4, c*4] to interleave features of 4 tokens
        x = x.view(b, num_tokens // 4, c * 4)

        # Apply the linear transformation
        x = self.linear(x)
        return x


class FullLinear(nn.Module):
    def __init__(self, config):
        super(FullLinear, self).__init__()
        self.linear = nn.Linear(config.mm_hidden_size, config.hidden_size)

    def forward(self, x):
        x = self.linear(x)
        return x

    @property
    def proj_out_num(self):
        num = 2048
        return num


class VolumeAggregator(nn.Module):
    def __init__(self, embed_dim=768, volume_dim=256):
        super().__init__()
        # Example: a simple Transformer encoder with a CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=1024
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Project to final volume-level embedding dimension
        self.volume_proj = nn.Linear(embed_dim, volume_dim)

    # def forward(self, slice_embeddings):
    #     """
    #     slice_embeddings: (S, E) or (batch_of_slices, E)
    #       In practice, we might pass them with shape (S, batch=1, E)
    #       to match Transformer input formatting.

    #     Returns: A single volume embedding of shape (volume_dim,).
    #     """

    #     B, S, E = slice_embeddings.shape  # Extract batch, slices, and embedding size

    #     # Add a CLS token at the beginning for each batch
    #     cls_token = self.cls_token.repeat(B, 1, 1)  # shape (B, 1, E)

    #     # Stack CLS + slice embeddings
    #     print(f"cls_token.shape: {cls_token.shape}")
    #     print(f"slice_embeddings.shpe: {slice_embeddings.shape}")

    #     tokens = torch.cat([cls_token, slice_embeddings], dim=0)  # (S+1, E)

    #     # For PyTorch Transformer, we need shape: (sequence_length, batch_size, embed_dim)
    #     tokens = tokens.unsqueeze(1)  # shape (S+1, 1, E)

    #     transformed = self.transformer_encoder(tokens)  # (S+1, 1, E)

    #     # Extract the CLS token output
    #     cls_output = transformed[0]  # shape (1, E)

    #     # Project to final dimension
    #     volume_embedding = self.volume_proj(cls_output)  # (1, volume_dim)
    #     volume_embedding = volume_embedding.squeeze(0)  # (volume_dim,)

    #     return volume_embedding
    def forward(self, slice_embeddings):
        """
        Forward pass for multi-batch processing.

        Args:
            slice_embeddings: Tensor of shape (B, S, E),
                            where B = batch size, S = number of slices, E = embedding size.

        Returns:
            volume_embedding: Tensor of shape (B, volume_dim).
        """
        B, S, E = slice_embeddings.shape  # Extract batch, slices, and embedding size

        # Add a CLS token at the beginning for each batch
        cls_token = self.cls_token.repeat(B, 1, 1)  # shape (B, 1, E)

        # Stack CLS token with slice embeddings along sequence dimension
        tokens = torch.cat([cls_token, slice_embeddings], dim=1)  # shape (B, S+1, E)

        # Reshape for transformer: (sequence_length, batch_size, embed_dim)
        tokens = tokens.permute(1, 0, 2)  # shape (S+1, B, E)

        # print(f"tokens.shape before transformer: {tokens.shape}")

        # Pass through the transformer encoder
        transformed = self.transformer_encoder(tokens)  # shape (S+1, B, E)

        # Extract the CLS token output (first token from the sequence)
        cls_output = transformed[0]  # shape (B, E)

        # Project to final volume embedding dimension
        volume_embedding = self.volume_proj(cls_output)  # shape (B, volume_dim)

        return volume_embedding  # shape (B, volume_dim)

    @property
    def proj_out_num(self):
        num = 256
        return num


def build_mm_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type")

    if projector_type == "linear":
        return FullLinear(config)

    elif projector_type == "spp":
        return SpatialPoolingProjector(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_dim=config.mm_hidden_size,
            out_dim=config.hidden_size,
            layer_type=config.proj_layer_type,
            layer_num=config.proj_layer_num,
            pooling_type=config.proj_pooling_type,
            pooling_size=config.proj_pooling_size,
        )

    elif projector_type == "identity":
        return IdentityMap()

    elif projector_type == "aggregator":
        return VolumeAggregator()
    else:
        raise ValueError(f"Unknown projector type: {projector_type}")
