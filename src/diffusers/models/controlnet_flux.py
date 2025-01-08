# -------------------------------------------------------------------------------------
# controlnet_flux.py (修正版2)
#  - FluxMultiControlNetModel維持
#  - residualにマスク適用する仕組み(例)を追加しつつ大幅削除はしない
# -------------------------------------------------------------------------------------
# Copyright 2024 ...
# Licensed under the Apache License, Version 2.0 ...

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


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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
            embedding_dim=self.inner_dim, 
            pooled_projection_dim=pooled_projection_dim
        )

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = torch.nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for i in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for i in range(num_single_layers)
            ]
        )

        # controlnet_blocks
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
        r"""
        Returns:
            `dict` of attention processors
        """
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
        count = len(self.attn_processors.keys())
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError("Mismatch in processor dict length")

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

    @classmethod
    def from_transformer(
        cls,
        transformer,
        num_layers: int = 4,
        num_single_layers: int = 10,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        load_weights_from_transformer=True,
    ):
        config = transformer.config
        config["num_layers"] = num_layers
        config["num_single_layers"] = num_single_layers
        config["attention_head_dim"] = attention_head_dim
        config["num_attention_heads"] = num_attention_heads

        controlnet = cls(**config)

        if load_weights_from_transformer:
            controlnet.pos_embed.load_state_dict(transformer.pos_embed.state_dict())
            controlnet.time_text_embed.load_state_dict(transformer.time_text_embed.state_dict())
            controlnet.context_embedder.load_state_dict(transformer.context_embedder.state_dict())
            controlnet.x_embedder.load_state_dict(transformer.x_embedder.state_dict())
            controlnet.transformer_blocks.load_state_dict(transformer.transformer_blocks.state_dict(), strict=False)
            controlnet.single_transformer_blocks.load_state_dict(
                transformer.single_transformer_blocks.state_dict(), strict=False
            )

            controlnet.controlnet_x_embedder = zero_module(controlnet.controlnet_x_embedder)

        return controlnet

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
        # (追加) mask_tensor: Optional[torch.Tensor] = None,
        mask_tensor: Optional[torch.Tensor] = None,  # (修正) 
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.
        (オリジナルに加え、mask_tensorでマスク外を residual=0 にする対応)
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        # add
        hidden_states = self.x_embedder(hidden_states)
        hidden_states = hidden_states + self.controlnet_x_embedder(controlnet_cond)

        if timestep is not None:
            timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

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

        # (warning for 3d txt_ids etc.)

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        block_samples = ()
        hidden_ = hidden_states
        enc_ = encoder_hidden_states
        for index_block, block in enumerate(self.transformer_blocks):
            enc_, hidden_ = block(
                hidden_states=hidden_,
                encoder_hidden_states=enc_,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
            block_samples = block_samples + (hidden_,)

        hidden_ = torch.cat([enc_, hidden_], dim=1)

        single_block_samples = ()
        for index_block, block in enumerate(self.single_transformer_blocks):
            hidden_ = block(
                hidden_states=hidden_,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
            single_block_samples = single_block_samples + (hidden_[:, enc_.shape[1]:],)

        # control blocks
        controlnet_block_samples = ()
        for bs_, cb_ in zip(block_samples, self.controlnet_blocks):
            out_ = cb_(bs_)
            controlnet_block_samples = controlnet_block_samples + (out_,)

        controlnet_single_block_samples = ()
        for sbs_, cbs_ in zip(single_block_samples, self.controlnet_single_blocks):
            out_2 = cbs_(sbs_)
            controlnet_single_block_samples = controlnet_single_block_samples + (out_2,)

        # (修正) mask_tensor があれば residual を mask内のみにする
        if mask_tensor is not None:
            # shape check: block_samples[i].(batch, seq_len, dim)
            #  mask_tensor ~ (batch, seq_len, 1)
            new_blocks = []
            for bb in controlnet_block_samples:
                new_blocks.append(bb * mask_tensor)
            controlnet_block_samples = tuple(new_blocks)

            new_single = []
            for sb in controlnet_single_block_samples:
                new_single.append(sb * mask_tensor)
            controlnet_single_block_samples = tuple(new_single)

        controlnet_block_samples = [bbb * conditioning_scale for bbb in controlnet_block_samples]
        controlnet_single_block_samples = [sss * conditioning_scale for sss in controlnet_single_block_samples]

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (controlnet_block_samples, controlnet_single_block_samples)

        return FluxControlNetOutput(
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
        )


class FluxMultiControlNetModel(ModelMixin):
    r"""
    `FluxMultiControlNetModel` wrapper class for Multi-FluxControlNetModel
    (オリジナルの docstring)
    """
    def __init__(self, controlnets):
        super().__init__()
        self.nets = nn.ModuleList(controlnets)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        controlnet_cond: List[torch.Tensor],
        controlnet_mode: List[torch.Tensor],
        conditioning_scale: List[float],
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        # (追加) mask_tensor
        mask_tensor: Optional[torch.Tensor] = None,
    ):
        # (オリジナルでは mask_tensor なし→最小限追加)
        if len(self.nets) == 1 and self.nets[0].union:
            controlnet = self.nets[0]

            for i, (image, mode, scale) in enumerate(zip(controlnet_cond, controlnet_mode, conditioning_scale)):
                block_samples, single_block_samples = controlnet(
                    hidden_states=hidden_states,
                    controlnet_cond=image,
                    controlnet_mode=mode[:, None],
                    conditioning_scale=scale,
                    timestep=timestep,
                    guidance=guidance,
                    pooled_projections=pooled_projections,
                    encoder_hidden_states=encoder_hidden_states,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                    mask_tensor=mask_tensor, # pass it
                )

                if i == 0:
                    control_block_samples = block_samples
                    control_single_block_samples = single_block_samples
                else:
                    control_block_samples = [
                        cbs + bs for cbs, bs in zip(control_block_samples, block_samples)
                    ]
                    control_single_block_samples = [
                        csbs + sbs for csbs, sbs in zip(control_single_block_samples, single_block_samples)
                    ]
        else:
            for i, (image, mode, scale, cn) in enumerate(zip(controlnet_cond, controlnet_mode, conditioning_scale, self.nets)):
                block_samples, single_block_samples = cn(
                    hidden_states=hidden_states,
                    controlnet_cond=image,
                    controlnet_mode=mode[:, None],
                    conditioning_scale=scale,
                    timestep=timestep,
                    guidance=guidance,
                    pooled_projections=pooled_projections,
                    encoder_hidden_states=encoder_hidden_states,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                    mask_tensor=mask_tensor,
                )
                if i == 0:
                    control_block_samples = block_samples
                    control_single_block_samples = single_block_samples
                else:
                    control_block_samples = [
                        cb + bs for cb, bs in zip(control_block_samples, block_samples)
                    ]
                    control_single_block_samples = [
                        cs + sbs for cs, sbs in zip(control_single_block_samples, single_block_samples)
                    ]

        if not return_dict:
            return (control_block_samples, control_single_block_samples)

        return FluxControlNetOutput(
            controlnet_block_samples=control_block_samples,
            controlnet_single_block_samples=control_single_block_samples,
        )
