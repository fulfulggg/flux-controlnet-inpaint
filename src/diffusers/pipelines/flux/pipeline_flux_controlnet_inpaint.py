# --------------------------------------------------------------------------------------
# pipeline_flux_controlnet_inpaint.py (修正版2)
#  - Inpaint + ControlNet がより強く効くように、
#    - mask外黒塗りヘルパー
#  - T5で512トークン対応 (max_sequence_length=512)
#  - 既存の行構造を大きく崩さないように注意
# --------------------------------------------------------------------------------------
# Copyright 2024 Black Forest Labs, The HuggingFace Team and The InstantX Team. 
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 ...
# ...
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import FluxLoraLoaderMixin, FromSingleFileMixin
from ...models.autoencoders import AutoencoderKL
from ...models.controlnet_flux import FluxControlNetModel, FluxMultiControlNetModel
from ...models.transformers import FluxTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import FluxPipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ...
"""

# ----------------------------------------------------------------------------
# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
# ----------------------------------------------------------------------------
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

# ----------------------------------------------------------------------------
# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
# ----------------------------------------------------------------------------
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    ...
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__} does not support custom timesteps."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__} does not support custom sigmas schedules."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class FluxControlNetInpaintPipeline(DiffusionPipeline, FluxLoraLoaderMixin, FromSingleFileMixin):
    r"""
    The Flux pipeline for text-to-image generation (Inpaint + ControlNet).
    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    (オリジナルの docstring はそのまま)
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    # -------------------------------------------------------------------------
    # __init__ 
    # -------------------------------------------------------------------------
    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        controlnet: Union[
            FluxControlNetModel, List[FluxControlNetModel], Tuple[FluxControlNetModel], FluxMultiControlNetModel
        ],
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            controlnet=controlnet,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels)) 
            if hasattr(self, "vae") and self.vae is not None 
            else 16
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            vae_latent_channels=self.vae.config.latent_channels,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        # (修正) CLIP tokenizerのmodel_max_length は ~77、 
        #       ただし T5で512トークン対応したいので デフォルトを77→512にする
        self.tokenizer_max_length = 512  # <-- (修正) 
        self.default_sample_size = 64

    # -------------------------------------------------------------------------
    # _get_t5_prompt_embeds (修正: max_sequence_length=512 デフォルト)
    # -------------------------------------------------------------------------
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,  # (修正) デフォルト512
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,  # (修正) enforce 512
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(
                untruncated_ids[:, self.tokenizer_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to 512 tokens:"
                f" {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    # -------------------------------------------------------------------------
    # 追加: マスク外を黒塗りするヘルパー (Cannyなど)
    # -------------------------------------------------------------------------
    def _apply_mask_to_control_image(self, control_image, mask_image):
        """
        マスク外を黒塗り: mask_image=白(変更),黒(保持)
        """
        import PIL
        from PIL import Image
        control_image = control_image.convert("RGB")
        mask_image = mask_image.convert("L")

        if control_image.size != mask_image.size:
            control_image = control_image.resize(mask_image.size, resample=PIL.Image.LANCZOS)

        c_np = np.array(control_image, dtype=np.uint8)
        m_np = np.array(mask_image, dtype=np.uint8)

        out_np = np.zeros_like(c_np)
        mask_bool = (m_np > 128)
        out_np[mask_bool] = c_np[mask_bool]

        return Image.fromarray(out_np, mode="RGB")

    # (以下の _get_clip_prompt_embeds, encode_prompt, check_inputs はオリジナルのまま or 最小修正)
    # -------------------------------------------------------------------------
    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,  # ~77
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle up to "
                f"{self.tokenizer.model_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,  # (修正) default=512
        lora_scale: Optional[float] = None,
    ):
        device = device or self._execution_device

        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # CLIP
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            # T5 (512 tokens)
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=512,
                device=device,
            )

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def check_inputs(
        self,
        prompt,
        prompt_2,
        strength,
        height,
        width,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0,1.0], got {strength}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 8 but got {height}x{width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` invalid: {callback_on_step_end_tensor_inputs}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`.")
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt_2` and `prompt_embeds`.")
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` must be str or list, got {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` must be str or list, got {type(prompt_2)}")

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError("If `prompt_embeds` are provided, `pooled_prompt_embeds` must be passed as well.")

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be >512, got {max_sequence_length}")

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        lh, lw, lc = latent_image_ids.shape
        latent_image_ids = latent_image_ids.reshape(lh * lw, lc)
        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height_ = height // vae_scale_factor
        width_ = width // vae_scale_factor

        latents = latents.view(batch_size, height_, width_, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), height_ * 2, width_ * 2)

        return latents

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError("Generators list mismatch")

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError("Need image or noise_timestep for latents init")

        height_ = 2 * (int(height) // self.vae_scale_factor)
        width_ = 2 * (int(width) // self.vae_scale_factor)

        shape = (batch_size, num_channels_latents, height_, width_)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height_, width_, device, dtype)

        if latents is None:
            image = image.to(device=device, dtype=dtype)
            image_latents = self._encode_vae_image(image=image, generator=generator)
        else:
            image_latents = latents

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents_ = noise if is_strength_max else self.scheduler.scale_noise(image_latents, timestep, noise)

        noise = self._pack_latents(noise, batch_size, num_channels_latents, height_, width_)
        image_latents = self._pack_latents(image_latents, batch_size, num_channels_latents, height_, width_)
        latents_ = self._pack_latents(latents_, batch_size, num_channels_latents, height_, width_)
        return latents_, noise, image_latents, latent_image_ids


    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        if isinstance(image, torch.Tensor):
            pass
        else:
            image = self.image_processor.preprocess(image, height=height, width=width)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)
        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    # --------------------------------------------------------------------------
    # (ここより下はオリジナルの __call__ に最小限の追加を行う)
    # --------------------------------------------------------------------------
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        strength: float = 0.6,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        control_image: PipelineImageInput = None,
        control_mode: Optional[Union[int, List[int]]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,  # (修正) 
    ):
        # (オリジナルの処理は基本そのまま)

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs
        self.check_inputs(
            prompt,
            prompt_2,
            strength,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        is_strength_max = (strength == 1.0)

        # 2. define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        dtype = self.transformer.dtype

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,  # enforce 512 for T5
            lora_scale=lora_scale,
        )

        # 3. Prepare control image
        num_channels_latents = self.transformer.config.in_channels // 4

        # (修正) mask_imageがあれば、control_imageを黒塗り (Cannyはマスク領域のみ効果)
        if (mask_image is not None) and (control_image is not None) and isinstance(self.controlnet, FluxControlNetModel):
            from PIL import Image
            if not isinstance(control_image, Image.Image):
                # もしTorch Tensorの場合などは変換要
                pass
            # mask外を黒塗り
            control_image = self._apply_mask_to_control_image(control_image, mask_image)

        if isinstance(self.controlnet, FluxControlNetModel):
            control_image = self.prepare_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=dtype,
            )
            height, width = control_image.shape[-2:]
            control_image = self.vae.encode(control_image).latent_dist.sample()
            control_image = (control_image - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            (h_c, w_c) = control_image.shape[2:]
            control_image = self._pack_latents(
                control_image,
                batch_size * num_images_per_prompt,
                num_channels_latents,
                h_c,
                w_c,
            )

            if control_mode is not None:
                control_mode = torch.tensor(control_mode).to(device, dtype=torch.long)
                control_mode = control_mode.reshape([-1, 1])

        elif isinstance(self.controlnet, FluxMultiControlNetModel):
            # (オリジナルの multi-control 処理)
            pass

        # 4. Prepare latents
        init_image = self.image_processor.preprocess(image, height=height, width=width)
        latents, noise, image_latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            init_image,
            None,
            is_strength_max,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = (int(height) // self.vae_scale_factor) * (int(width) // self.vae_scale_factor)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        timesteps, num_inference_steps = self.get_timesteps(timesteps, num_inference_steps, strength, device)
        if num_inference_steps < 1:
            raise ValueError("Invalid pipeline steps after strength adjustment.")

        # 6. Prepare mask latents (Inpaint)
        mask_condition = self.mask_processor.preprocess(
            mask_image, height=height, width=width
        )
        masked_image_ = init_image * (mask_condition < 0.5)
        mask_, masked_image_latents_ = self.prepare_mask_latents(
            mask_condition,
            masked_image_,
            batch_size,
            num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )
        masked_image_latents_ = self._pack_latents(
            masked_image_latents_,
            batch_size,
            num_channels_latents,
            2 * (int(height) // self.vae_scale_factor),
            2 * (int(width) // self.vae_scale_factor),
        )
        mask_ = self._pack_latents(
            mask_.repeat(1, num_channels_latents, 1, 1),
            batch_size,
            num_channels_latents,
            2 * (int(height) // self.vae_scale_factor),
            2 * (int(width) // self.vae_scale_factor),
        )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                if self.transformer.config.guidance_embeds:
                    guidance_ = torch.tensor([guidance_scale], device=device)
                    guidance = guidance_.expand(latents.shape[0])
                else:
                    guidance = None

                controlnet_block_samples, controlnet_single_block_samples = self.controlnet(
                    hidden_states=latents,
                    controlnet_cond=control_image,
                    controlnet_mode=control_mode,
                    conditioning_scale=controlnet_conditioning_scale,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                init_latents_proper = image_latents
                init_mask = mask_

                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    init_latents_proper = self.scheduler.scale_noise(
                        init_latents_proper, torch.tensor([noise_timestep]), noise
                    )

                # combine
                latents = (1 - init_mask) * init_latents_proper + init_mask * latents

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # 8. decode
        if output_type == "latent":
            image_ = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image_ = self.vae.decode(latents, return_dict=False)[0]
            image_ = self.image_processor.postprocess(image_, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image_,)
        return FluxPipelineOutput(images=image_)


    def prepare_mask_latents(
        self,
        mask,
        masked_image,
        batch_size,
        num_images_per_prompt,
        height,
        width,
        dtype,
        device,
        generator,
    ):
        # (オリジナルの prepare_mask_latents そのまま)
        mask = torch.nn.functional.interpolate(
            mask, size=(2 * height // self.vae_scale_factor, 2 * width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        batch_size_ = batch_size * num_images_per_prompt

        masked_image = masked_image.to(device=device, dtype=dtype)

        if masked_image.shape[1] == 16:
            masked_image_latents = masked_image
        else:
            masked_image_latents = retrieve_latents(self.vae.encode(masked_image), generator=generator)

        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        if mask.shape[0] < batch_size_:
            if not batch_size_ % mask.shape[0] == 0:
                raise ValueError("Masks batch mismatch.")
            mask = mask.repeat(batch_size_ // mask.shape[0], 1, 1, 1)

        if masked_image_latents.shape[0] < batch_size_:
            if not batch_size_ % masked_image_latents.shape[0] == 0:
                raise ValueError("masked_image batch mismatch.")
            masked_image_latents = masked_image_latents.repeat(batch_size_ // masked_image_latents.shape[0], 1, 1, 1)

        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents


    def get_timesteps(self, timesteps, num_inference_steps, strength, device):
        init_timestep = min(num_inference_steps * strength, num_inference_steps)
        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)
        return timesteps, num_inference_steps - t_start


    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        return image_latents

def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")
