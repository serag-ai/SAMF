from .vit import ViT3DTower
from .dino_vit import Dino_vitb16


def build_vision_tower(config, **kwargs):
    vision_tower = getattr(config, "vision_tower", None)
    if "vit3d" in vision_tower.lower():
        return ViT3DTower(config, **kwargs)
    if "dino_large" in vision_tower.lower():
        return Dino_vitb16()
    else:
        raise ValueError(f"Unknown vision tower: {vision_tower}")
