# -------------------------------------------------------------------------------------
# controlnet_flux.py (修正版)
# -------------------------------------------------------------------------------------
# Copyright 2024 Black Forest Labs...
# Licensed under the Apache License, Version 2.0

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..loaders import PeftAdapterMixin
from ..models.attention_processor import AttentionProcessor
from ..models.modeling_utils import ModelMixin
from ..utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from .controlnet import BaseOutput, zero_module
from .embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings, FluxPosEmbed
from .modeling_outputs import Transformer2DModelOutput
from .transformers.transformer_flux import FluxSingleTransformerBlock, FluxTransformerBlock

logger = logging.get_logger(__name__)

@dataclass
class FluxControlNetOutput(BaseOutput):
    controlnet_block_samples: Tuple[torch.Tensor]
    controlnet_single_block_samples: Tuple[torch.Tensor]


class FluxControlNetModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: List[int] = [16, 56, 56],
        num_mode: int = None,
    ):
        super().__init__()
        self.out_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)
        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = torch.nn.Linear(in_channels, self.inner_dim)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_single_layers)
            ]
        )

        # controlnet blocks
        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(len(self.transformer_blocks)):
            self.controlnet_blocks.append(zero_module(nn.Linear(self.inner_dim, self.inner_dim)))

        self.controlnet_single_blocks = nn.ModuleList([])
        for _ in range(len(self.single_transformer_blocks)):
            self.controlnet_single_blocks.append(zero_module(nn.Linear(self.inner_dim, self.inner_dim)))

        self.union = (num_mode is not None)
        if self.union:
            self.controlnet_mode_embedder = nn.Embedding(num_mode, self.inner_dim)

        self.controlnet_x_embedder = zero_module(torch.nn.Linear(in_channels, self.inner_dim))

        self.gradient_checkpointing = False

    @property
    def attn_processors(self):
        # (既存)
        processors = {}
        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()
            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
            return processors
        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)
        return processors

    def set_attn_processor(self, processor):
        # (既存)
        count = len(self.attn_processors.keys())
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError("dict of processors mismatch")
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)
        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        controlnet_mode: torch.Tensor = None,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        # 追加
        mask_tensor: Optional[torch.Tensor] = None,  # マスク適用用
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:

        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        hidden_states = self.x_embedder(hidden_states)

        # control_image 加算
        hidden_states = hidden_states + self.controlnet_x_embedder(controlnet_cond)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        if guidance is None:
            temb = self.time_text_embed(timestep, pooled_projections)
        else:
            temb = self.time_text_embed(timestep, guidance, pooled_projections)

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if self.union:
            if controlnet_mode is None:
                raise ValueError("`controlnet_mode` cannot be None in union")
            controlnet_mode_emb = self.controlnet_mode_embedder(controlnet_mode)
            encoder_hidden_states = torch.cat([controlnet_mode_emb, encoder_hidden_states], dim=1)
            txt_ids = torch.cat([txt_ids[:1], txt_ids], dim=0)

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        block_samples = ()
        for index_block, block in enumerate(self.transformer_blocks):
            # (既存) gradient_checkpointing etc
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
            block_samples = block_samples + (hidden_states,)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        single_block_samples = ()
        for index_block, block in enumerate(self.single_transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
            single_block_samples = single_block_samples + (
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

        # controlnet_block
        controlnet_block_samples = ()
        for block_sample, controlnet_block in zip(block_samples, self.controlnet_blocks):
            block_sample = controlnet_block(block_sample)
            controlnet_block_samples = controlnet_block_samples + (block_sample,)

        controlnet_single_block_samples = ()
        for single_block_sample, controlnet_block in zip(single_block_samples, self.controlnet_single_blocks):
            single_block_sample = controlnet_block(single_block_sample)
            controlnet_single_block_samples = controlnet_single_block_samples + (single_block_sample,)

        # マスク適用 (outside mask -> 0)
        if mask_tensor is not None:
            # controlnet_block_samples[i].shape: (batch, seq_len, dim) 
            # mask_tensor shape must match
            masked_block_samples = []
            for sample in controlnet_block_samples:
                masked_block_samples.append(sample * mask_tensor)
            controlnet_block_samples = tuple(masked_block_samples)

            masked_single_block = []
            for sample in controlnet_single_block_samples:
                masked_single_block.append(sample * mask_tensor)
            controlnet_single_block_samples = tuple(masked_single_block)

        # scale
        controlnet_block_samples = [s * conditioning_scale for s in controlnet_block_samples]
        controlnet_single_block_samples = [s * conditioning_scale for s in controlnet_single_block_samples]

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (controlnet_block_samples, controlnet_single_block_samples)

        return FluxControlNetOutput(
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
        )
