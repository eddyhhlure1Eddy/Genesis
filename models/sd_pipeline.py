"""
Genesis Stable Diffusion Pipeline
Simplified SD inference wrapper using diffusers
Author: eddy
"""

import torch
from typing import Optional, Dict, Any, Callable
import logging
from PIL import Image
import numpy as np


class StableDiffusionPipeline:
    """
    Stable Diffusion Pipeline wrapper

    Provides simplified interface for SD inference
    """

    def __init__(self, device: torch.device):
        """
        Initialize SD Pipeline

        Args:
            device: Computing device
        """
        self.device = device
        self.logger = logging.getLogger('Genesis.SDPipeline')

        self.model = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.unet = None
        self.scheduler = None

        self._initialized = False

    def load_from_state_dict(
        self,
        checkpoint_state_dict: Dict[str, torch.Tensor],
        vae_state_dict: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Load models from state dicts

        Args:
            checkpoint_state_dict: Checkpoint state dict
            vae_state_dict: Optional VAE state dict
        """
        try:
            from diffusers import (
                AutoencoderKL,
                UNet2DConditionModel,
                DDPMScheduler,
                PNDMScheduler,
                LMSDiscreteScheduler,
                EulerDiscreteScheduler,
                EulerAncestralDiscreteScheduler,
                DPMSolverMultistepScheduler
            )
            from transformers import CLIPTextModel, CLIPTokenizer
        except ImportError as e:
            self.logger.error(f"Missing dependencies: {e}")
            self.logger.error("Install: pip install diffusers transformers")
            raise

        self.logger.info("Initializing Stable Diffusion components...")

        self.logger.info("Loading VAE...")
        if vae_state_dict:
            self.vae = AutoencoderKL.from_config({
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 4,
                "block_out_channels": [128, 256, 512, 512],
            })
            self.vae.load_state_dict(vae_state_dict)
        else:
            self.vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse",
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            )

        self.vae = self.vae.to(self.device)
        self.vae.eval()

        self.logger.info("Loading Text Encoder and Tokenizer...")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
        )
        self.text_encoder = self.text_encoder.to(self.device)
        self.text_encoder.eval()

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        self.logger.info("Loading UNet...")
        self.unet = UNet2DConditionModel.from_config({
            "in_channels": 4,
            "out_channels": 4,
            "cross_attention_dim": 768,
            "attention_head_dim": 8,
            "block_out_channels": [320, 640, 1280, 1280],
        })

        self.unet.load_state_dict(checkpoint_state_dict, strict=False)
        self.unet = self.unet.to(self.device)
        self.unet.eval()

        self.scheduler = EulerDiscreteScheduler.from_config({
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear"
        })

        self._initialized = True
        self.logger.info("✓ SD Pipeline initialized")

    def load_from_pretrained(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        """
        Load pre-trained model from HuggingFace

        Args:
            model_id: Model ID on HuggingFace
        """
        try:
            from diffusers import StableDiffusionPipeline as HFPipeline
        except ImportError as e:
            self.logger.error(f"Missing diffusers: {e}")
            raise

        self.logger.info(f"Loading pre-trained model: {model_id}")

        pipeline = HFPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            safety_checker=None
        )

        self.vae = pipeline.vae.to(self.device)
        self.text_encoder = pipeline.text_encoder.to(self.device)
        self.tokenizer = pipeline.tokenizer
        self.unet = pipeline.unet.to(self.device)
        self.scheduler = pipeline.scheduler

        self.vae.eval()
        self.text_encoder.eval()
        self.unet.eval()

        self._initialized = True
        self.logger.info("✓ Pre-trained model loaded")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: Optional[int] = None,
        callback: Optional[Callable] = None
    ) -> Image.Image:
        """
        Generate image from text prompt

        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            width: Image width
            height: Image height
            steps: Number of inference steps
            cfg_scale: Guidance scale
            seed: Random seed
            callback: Progress callback(step, total_steps)

        Returns:
            Generated PIL Image
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call load_from_pretrained() or load_from_state_dict() first")

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        batch_size = 1

        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        uncond_input = self.tokenizer(
            [negative_prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, height // 8, width // 8),
            device=self.device,
            dtype=text_embeddings.dtype
        )

        self.scheduler.set_timesteps(steps)
        latents = latents * self.scheduler.init_noise_sigma

        for i, t in enumerate(self.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            if callback:
                callback(i + 1, steps)

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image[0] * 255).astype(np.uint8)

        return Image.fromarray(image)

    def cleanup(self):
        """Cleanup resources"""
        if self.vae:
            del self.vae
        if self.text_encoder:
            del self.text_encoder
        if self.unet:
            del self.unet

        self.vae = None
        self.text_encoder = None
        self.unet = None
        self.scheduler = None
        self._initialized = False

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
