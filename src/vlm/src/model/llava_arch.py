from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_mm_projector
from .multimodal_fuse.builder import build_mm_fusion


class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        self.config = config

        if hasattr(config, "vision_tower"):
            self.vision_tower = build_vision_tower(config)
            self.mm_projector = build_mm_projector(config)
            self.mm_fusion = build_mm_fusion(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        return vision_tower

    def initialize_vision_modules(self, model_args):
        self.config.image_channel = model_args.image_channel
        self.config.image_size = model_args.image_size
        self.config.patch_size = model_args.patch_size

        self.config.vision_tower = model_args.vision_tower
        self.config.vision_select_layer = model_args.vision_select_layer
        self.config.vision_select_feature = model_args.vision_select_feature

        self.config.mm_projector_type = model_args.mm_projector_type
        self.config.proj_layer_type = model_args.proj_layer_type
        self.config.proj_layer_num = model_args.proj_layer_num
        self.config.proj_pooling_type = model_args.proj_pooling_type
        self.config.proj_pooling_size = model_args.proj_pooling_size

        self.config.mm_fuse_type = model_args.mm_fuse_type

        # vision tower
        if self.get_vision_tower() is None:
            self.vision_tower = build_vision_tower(self.config)
            self.vision_tower.requires_grad_(not model_args.freeze_vision_tower)

        if model_args.pretrain_vision_model is not None:
            vision_model_weights = torch.load(
                model_args.pretrain_vision_model, map_location="cpu"
            )
            self.vision_tower.vision_tower.load_state_dict(
                vision_model_weights, strict=True
            )

        self.config.mm_hidden_size = self.vision_tower.hidden_size

        # mm_projector
        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_mm_projector(self.config)
            self.mm_projector.requires_grad_(not model_args.freeze_aggregator)

        if model_args.pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                model_args.pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.mm_projector.load_state_dict(mm_projector_weights)

        if getattr(self, "mm_fusion", None) is None:
            self.mm_fusion = build_mm_fusion(self.config)

        if model_args.pretrain_mm_fusion is not None:
            mm_projector_weights = torch.load(
                model_args.pretrain_mm_fusion, map_location="cpu"
            )

            self.mm_projector.load_state_dict(mm_projector_weights)


class LlavaMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images, input_embed):

        feat_2d = self.get_model().get_vision_tower()(images)
        feat_3d = self.get_model().mm_projector(feat_2d)

        image_features = self.get_model().mm_fusion(
            feat_3d=feat_3d, feat_2d=feat_2d, feat_text=input_embed
        )

        return image_features

    def prepare_inputs_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            image_features = self.encode_images(images, inputs_embeds)

            inputs_embeds = torch.cat(
                (
                    inputs_embeds[:, :1, :],
                    image_features,
                    inputs_embeds[:, (image_features.shape[1] + 1) :, :],
                ),
                dim=1,
            )

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        )

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        num_new_tokens = model_args.num_new_tokens

        self.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
            else:
                # we add 4 new tokens
                # if new tokens need input, please train input_embeddings
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                # if new tokens need predict, please train output_embeddings
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = True

        if model_args.pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(
                model_args.pretrain_mm_mlp_adapter, map_location="cpu"
            )
