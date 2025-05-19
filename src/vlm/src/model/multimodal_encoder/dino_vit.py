import torch
import torch.nn as nn


class Dino_vitb16(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = torch.hub.load(
            "facebookresearch/dino:main", "dino_vitb16", pretrained=False
        )

    def forward(self, images):
        """
        Forward pass for batch processing.

        Args:
            images: Tensor of shape (B, S, C, H, W), where
                    B = batch size, S = number of slices, C = channels, H = height, W = width.

        Returns:
            slice_embeddings: Tensor of shape (B, S, E), where E is the embedding size.
        """
        batch_size, num_slices, _, _, _ = images.shape  # Extract dimensions
        slice_embeddings = []

        for batch_idx in range(batch_size):  # Iterate over batch
            batch_slices = images[batch_idx]  # Shape: (S, C, H, W)
            batch_embeddings = []

            for slice_image in batch_slices:  # Iterating over (C, H, W) slices
                slice_tensor = slice_image.unsqueeze(0).to(
                    self.device
                )  # Shape: (1, C, H, W)

                z_2d = self.vision_tower(slice_tensor)  # Output shape: (1, E)
                batch_embeddings.append(z_2d.squeeze(0))  # Shape: (E,)

            batch_embeddings = torch.stack(batch_embeddings, dim=0)  # Shape: (S, E)
            slice_embeddings.append(batch_embeddings)

        slice_embeddings = torch.stack(slice_embeddings, dim=0)  # Shape: (B, S, E)

        return slice_embeddings

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return torch.device("cuda")  # self.vision_tower.device

    @property
    def hidden_size(self):
        return self.vision_tower.num_features
