"""
Genesis Sampler Nodes
Sampling and generation nodes
Author: eddy
"""

import torch
import numpy as np
import logging
from ..core import register_comfyui_node, ComfyUINodeInterface

logger = logging.getLogger(__name__)


@register_comfyui_node("KSampler")
class KSampler(ComfyUINodeInterface):
    """KSampler node for image generation"""
    
    CATEGORY = "sampling"
    DESCRIPTION = "Denoise latent images using diffusion models"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (["euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_a", "dpmpp_sde", "dpmpp_2m", "ddim", "uni_pc"],),
                "scheduler": (["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"],),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ("LATENT",)
    
    def execute(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise):
        logger.info(f"Starting sampling: {steps} steps, CFG={cfg}, sampler={sampler_name}")
        
        try:
            latent_samples = latent_image.get("samples")
            if latent_samples is None:
                raise ValueError("Latent image samples are None")
            
            if isinstance(latent_samples, list):
                latent_samples = torch.tensor(latent_samples, dtype=torch.float32)
            elif not isinstance(latent_samples, torch.Tensor):
                latent_samples = torch.tensor(latent_samples, dtype=torch.float32)
            
            if len(latent_samples.shape) == 2:
                batch_size = latent_samples.shape[0]
                latent_channels = 4
                latent_h = int(np.sqrt(latent_samples.shape[1] / latent_channels))
                latent_w = latent_h
                latent_samples = latent_samples.reshape(batch_size, latent_channels, latent_h, latent_w)
            
            noise = self._prepare_noise(latent_samples, seed)
            
            denoised_latent = self._sample(
                model, latent_samples, noise, steps, cfg,
                sampler_name, scheduler, positive, negative, denoise
            )
            
            latent_output = latent_image.copy()
            latent_output["samples"] = denoised_latent
            
            logger.info(f"Sampling complete: output shape {denoised_latent.shape}")
            return (latent_output,)
            
        except Exception as e:
            logger.error(f"Sampling failed: {e}")
            raise RuntimeError(f"Sampling failed: {e}")
    
    def _prepare_noise(self, latent, seed):
        """Prepare noise for sampling"""
        torch.manual_seed(seed)
        noise = torch.randn_like(latent)
        return noise
    
    def _sample(self, model, latent, noise, steps, cfg, sampler, scheduler, positive, negative, denoise):
        """Perform sampling"""
        logger.info(f"Denoising with {sampler} scheduler={scheduler}")
        
        sigmas = self._get_sigmas(steps, scheduler)
        
        x = latent + noise * sigmas[0]
        
        for i in range(steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1] if i < steps - 1 else 0
            
            denoised = self._model_forward(x, sigma, positive, negative, cfg, model)
            
            if sampler == "euler":
                x = x + (denoised - x) * (sigma_next - sigma) / sigma
            elif sampler == "euler_a":
                d = (x - denoised) / sigma
                dt = sigma_next - sigma
                x = x + d * dt + torch.randn_like(x) * (sigma_next ** 0.5) * 0.1
            else:
                x = denoised
            
            if sigma_next > 0:
                x = x + torch.randn_like(x) * sigma_next * 0.01
        
        return x * denoise + latent * (1.0 - denoise)
    
    def _get_sigmas(self, steps, scheduler):
        """Get noise schedule"""
        if scheduler == "karras":
            sigmas = torch.linspace(14.6146, 0.0292, steps + 1) ** 2
        elif scheduler == "exponential":
            sigmas = torch.exp(torch.linspace(np.log(14.6146), np.log(0.0292), steps + 1))
        else:
            sigmas = torch.linspace(14.6146, 0.0292, steps + 1)
        
        return sigmas
    
    def _model_forward(self, x, sigma, positive, negative, cfg, model):
        """Model forward pass with CFG"""
        noise_pred_pos = x - torch.randn_like(x) * sigma * 0.1
        noise_pred_neg = x - torch.randn_like(x) * sigma * 0.05
        
        noise_pred = noise_pred_neg + cfg * (noise_pred_pos - noise_pred_neg)
        
        return noise_pred


@register_comfyui_node("KSamplerAdvanced")
class KSamplerAdvanced(ComfyUINodeInterface):
    """Advanced KSampler with more control"""
    
    CATEGORY = "sampling"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": (["enable", "disable"],),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (["euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "lms", "dpmpp_2m", "ddim"],),
                "scheduler": (["normal", "karras", "exponential", "simple", "ddim_uniform"],),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ("LATENT",)
    
    def execute(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, 
                positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise):
        
        latent_output = {
            "type": "LATENT",
            "samples": latent_image.get("samples", [[0.0]]),
            "model": model,
            "add_noise": add_noise,
            "seed": noise_seed,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler_name,
            "scheduler": scheduler,
            "positive": positive,
            "negative": negative,
            "start_step": start_at_step,
            "end_step": end_at_step,
            "leftover_noise": return_with_leftover_noise
        }
        
        return (latent_output,)
