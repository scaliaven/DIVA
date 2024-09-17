import clip
import torch.nn as nn
import torch
from open_clip import create_model_from_pretrained, create_model_and_transforms
from collections import OrderedDict
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
from transformers.trainer import logger
from .utils import CombinedModel

def load_map_state_dict(path):

    state_dict = dict(torch.load(path)) # state dict
    mapped_state_dict = {} # mapped state dict

    mapped_state_dict["vision_model.embeddings.class_embedding"] = \
        state_dict["visual.class_embedding"]
    # mapped_state_dict["vision_model.embeddings.patch_embedding.weight"] = \
    #     state_dict["token_embedding.weight"]
    # mapped_state_dict["vision_model.embeddings.position_embedding.weight"] = \
    #     state_dict["positional_embedding"]
    mapped_state_dict["vision_model.pre_layrnorm.weight"] = \
        state_dict["visual.ln_pre.weight"]
    mapped_state_dict["vision_model.pre_layrnorm.bias"] = \
        state_dict["visual.ln_pre.bias"]
    mapped_state_dict["vision_model.post_layernorm.weight"] = \
        state_dict["visual.ln_pre.bias"]
    mapped_state_dict["vision_model.post_layernorm.bias"] = \
        state_dict["visual.ln_pre.bias"]
    mapped_state_dict["visual_projection.weight"] = \
        state_dict["visual.proj"].T

    for i in range(0, 24):
        mapped_state_dict[f"vision_model.encoder.layers.{i}.self_attn.k_proj.weight"] = \
            state_dict[f"visual.transformer.resblocks.{i}.attn.in_proj_weight"][:1024, :]
        mapped_state_dict[f"vision_model.encoder.layers.{i}.self_attn.k_proj.bias"] = \
            state_dict[f"visual.transformer.resblocks.{i}.attn.in_proj_bias"][:1024]
        mapped_state_dict[f"vision_model.encoder.layers.{i}.self_attn.v_proj.weight"] = \
            state_dict[f"visual.transformer.resblocks.{i}.attn.in_proj_weight"][1024:2048, :]
        mapped_state_dict[f"vision_model.encoder.layers.{i}.self_attn.v_proj.bias"] = \
            state_dict[f"visual.transformer.resblocks.{i}.attn.in_proj_bias"][1024:2048]
        mapped_state_dict[f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight"] = \
            state_dict[f"visual.transformer.resblocks.{i}.attn.in_proj_weight"][2048:3072, :]
        mapped_state_dict[f"vision_model.encoder.layers.{i}.self_attn.q_proj.bias"] = \
            state_dict[f"visual.transformer.resblocks.{i}.attn.in_proj_bias"][2048:3072]
        mapped_state_dict[f"vision_model.encoder.layers.{i}.self_attn.out_proj.weight"] = \
            state_dict[f"visual.transformer.resblocks.{i}.attn.out_proj.weight"]
        mapped_state_dict[f"vision_model.encoder.layers.{i}.self_attn.out_proj.bias"] = \
            state_dict[f"visual.transformer.resblocks.{i}.attn.out_proj.bias"]
        mapped_state_dict[f"vision_model.encoder.layers.{i}.layer_norm1.weight"] = \
            state_dict[f"visual.transformer.resblocks.{i}.ln_1.weight"]
        mapped_state_dict[f"vision_model.encoder.layers.{i}.layer_norm1.bias"] = \
            state_dict[f"visual.transformer.resblocks.{i}.ln_1.bias"]
        mapped_state_dict[f"vision_model.encoder.layers.{i}.mlp.fc1.weight"] = \
            state_dict[f"visual.transformer.resblocks.{i}.mlp.c_fc.weight"]
        mapped_state_dict[f"vision_model.encoder.layers.{i}.mlp.fc1.bias"] = \
            state_dict[f"visual.transformer.resblocks.{i}.mlp.c_fc.bias"]
        mapped_state_dict[f"vision_model.encoder.layers.{i}.mlp.fc2.weight"] = \
            state_dict[f"visual.transformer.resblocks.{i}.mlp.c_proj.weight"]
        mapped_state_dict[f"vision_model.encoder.layers.{i}.mlp.fc2.bias"] = \
            state_dict[f"visual.transformer.resblocks.{i}.mlp.c_proj.bias"]
        mapped_state_dict[f"vision_model.encoder.layers.{i}.layer_norm2.weight"] = \
            state_dict[f"visual.transformer.resblocks.{i}.ln_2.weight"]
        mapped_state_dict[f"vision_model.encoder.layers.{i}.layer_norm2.bias"] = \
            state_dict[f"visual.transformer.resblocks.{i}.ln_2.bias"]

    return OrderedDict(mapped_state_dict)


class OpenAICLIP(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.clip_image_size == 224:

            clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
            # model, _ = clip.load("ViT-L/14", jit=False)
            # state_dict = torch.load('pretrained_weights/CLIP/OpenAI-ViT-L-14-224.pth')
            state_dict = load_map_state_dict('pretrained_weights/CLIP/OpenAI-ViT-L-14-224.pth')
            clip_model.load_state_dict(state_dict, strict=False)
            word_embedding = nn.Embedding(49408, 256) # 256 is the size of the word embedding
            model = CombinedModel(clip_model, word_embedding)

        if config.clip_image_size == 336:
            model, _ = clip.load("pretrained_weights/CLIP/ViT-L-14-336px.pt",jit=False)

        self.final_fc = nn.Linear(256, config.actual_bs, bias=False)
        self.model = model
        self.config = config

    def forward(self, images):
        
        image_features = self.model.encode_image(images).float()
        logits = 100. * self.final_fc(image_features[:,0,:]).float()

        return image_features, logits


class DFN(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.clip_image_size == 224:
            model, _ = create_model_from_pretrained(model_name='ViT-H-14-quickgelu', pretrained="pretrained_weights/CLIP/DFN5B-CLIP-ViT-H-14/open_clip_pytorch_model.bin")
        if config.clip_image_size == 378:
            model, _ = create_model_from_pretrained(model_name='ViT-H-14-378-quickgelu', pretrained="pretrained_weights/CLIP/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin")
        
        self.final_fc = nn.Linear(1024, config.actual_bs, bias=False)
        self.model = model
        self.config = config

    def forward(self, images):
        
        image_features = self.model.encode_image(images).float()
        logits = 100. * self.final_fc(image_features[:,0,:]).float()    

        return image_features, logits
    
    
class SigLIP(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.clip_image_size == 224:
            model, _ = create_model_from_pretrained(model_name='ViT-SO400M-14-SigLIP', pretrained="pretrained_weights/CLIP/ViT-SO400M-14-SigLIP/open_clip_pytorch_model.bin",
                                                    image_mean=([0.5,0.5,0.5]), image_std=([0.5,0.5,0.5]), image_interpolation="bicubic", image_resize_mode="squash")
        if config.clip_image_size == 384:
            model, _ = create_model_from_pretrained(model_name='ViT-SO400M-14-SigLIP-384', pretrained="pretrained_weights/CLIP/ViT-SO400M-14-SigLIP-384/open_clip_pytorch_model.bin",
                                                     image_mean=([0.5,0.5,0.5]), image_std=([0.5,0.5,0.5]), image_interpolation="bicubic", image_resize_mode="squash")

        self.final_fc = nn.Linear(1152, config.actual_bs, bias=False)
        self.model = model
        self.config = config

    def forward(self, images):
        
        image_features = self.model.encode_image(images).float()
        logits = 100. * self.final_fc(image_features[:,0,:]).float()    

        return image_features, logits


class MetaCLIP(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.metaclip_version == "large":
            model, _, _ = create_model_and_transforms(model_name='ViT-L-14-quickgelu', pretrained="pretrained_weights/CLIP/MetaCLIP/l14_fullcc2.5b.pt")
            self.final_fc = nn.Linear(768, config.actual_bs, bias=False)
        if config.metaclip_version == "huge":
            model, _, _ = create_model_and_transforms(model_name='ViT-H-14-quickgelu', pretrained="pretrained_weights/CLIP/MetaCLIP/h14_fullcc2.5b.pt")
            self.final_fc = nn.Linear(1024, config.actual_bs, bias=False)

        self.model = model
        self.config = config

    def forward(self, images):
        
        image_features = self.model.encode_image(images).float()
        logits = 100. * self.final_fc(image_features[:,0,:]).float()    

        return image_features, logits
