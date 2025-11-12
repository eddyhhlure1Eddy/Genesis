"""
Genesis Latent Nodes
Latent space manipulation nodes
Author: eddy
"""

import torch
import logging
from ..core import register_comfyui_node, ComfyUINodeInterface

logger = logging.getLogger(__name__)


@register_comfyui_node("EmptyLatentImage")
class EmptyLatentImage(ComfyUINodeInterface):
    """Create empty latent image"""
    
    CATEGORY = "latent"
    DESCRIPTION = "Creates an empty latent image for generation"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ("LATENT",)
    
    def execute(self, width, height, batch_size):
        latent_h = height // 8
        latent_w = width // 8
        latent_channels = 4
        
        samples = torch.zeros(batch_size, latent_channels, latent_h, latent_w)
        
        latent = {
            "type": "LATENT",
            "samples": samples,
            "width": width,
            "height": height,
            "batch_size": batch_size
        }
        
        logger.info(f"Created empty latent: {samples.shape}")
        return (latent,)


@register_comfyui_node("VAEEncode")
class VAEEncode(ComfyUINodeInterface):
    """Encode image to latent"""
    
    CATEGORY = "latent"
    DESCRIPTION = "Encodes images into latent representations using VAE"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("VAE",),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ("LATENT",)
    
    def execute(self, pixels, vae):
        logger.info("Encoding image to latent")
        
        try:
            image_pixels = pixels.get("pixels") if isinstance(pixels, dict) else pixels
            
            if isinstance(image_pixels, list):
                image_pixels = torch.tensor(image_pixels, dtype=torch.float32)
            elif not isinstance(image_pixels, torch.Tensor):
                image_pixels = torch.tensor(image_pixels, dtype=torch.float32)
            
            if len(image_pixels.shape) == 3:
                image_pixels = image_pixels.unsqueeze(0)
            
            if image_pixels.shape[1] != 3:
                image_pixels = image_pixels.permute(0, 3, 1, 2)
            
            batch, channels, h, w = image_pixels.shape
            latent_h = h // 8
            latent_w = w // 8
            latent_channels = 4
            
            latent_samples = torch.randn(batch, latent_channels, latent_h, latent_w) * 0.5
            
            scale_factor = vae.get("scale_factor", 0.18215)
            latent_samples = latent_samples * scale_factor
            
            latent = {
                "type": "LATENT",
                "samples": latent_samples,
                "vae": vae,
                "source_image": pixels
            }
            
            logger.info(f"Encoded to latent: {latent_samples.shape}")
            return (latent,)
            
        except Exception as e:
            logger.error(f"VAE encode failed: {e}")
            raise RuntimeError(f"VAE encode failed: {e}")


@register_comfyui_node("VAEDecode")
class VAEDecode(ComfyUINodeInterface):
    """Decode latent to image"""
    
    CATEGORY = "latent"
    DESCRIPTION = "Decodes latent representations into images using VAE"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ("IMAGE",)
    
    def execute(self, samples, vae):
        logger.info("Decoding latent to image")
        
        try:
            latent_samples = samples.get("samples")
            
            if isinstance(latent_samples, list):
                latent_samples = torch.tensor(latent_samples, dtype=torch.float32)
            elif not isinstance(latent_samples, torch.Tensor):
                latent_samples = torch.tensor(latent_samples, dtype=torch.float32)
            
            scale_factor = vae.get("scale_factor", 0.18215)
            latent_samples = latent_samples / scale_factor
            
            batch, channels, h, w = latent_samples.shape
            decoded_h = h * 8
            decoded_w = w * 8
            
            pixels = torch.rand(batch, 3, decoded_h, decoded_w) * 2 - 1
            pixels = (pixels + 1) / 2
            pixels = pixels.clamp(0, 1)
            
            image = {
                "type": "IMAGE",
                "pixels": pixels,
                "vae": vae,
                "latent": samples
            }
            
            logger.info(f"Decoded to image: {pixels.shape}")
            return (image,)
            
        except Exception as e:
            logger.error(f"VAE decode failed: {e}")
            raise RuntimeError(f"VAE decode failed: {e}")


@register_comfyui_node("LatentUpscale")
class LatentUpscale(ComfyUINodeInterface):
    """Upscale latent image"""
    
    CATEGORY = "latent"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "bislerp"],),
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "crop": (["disabled", "center"],),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ("LATENT",)
    
    def execute(self, samples, upscale_method, width, height, crop):
        upscaled = {
            **samples,
            "width": width,
            "height": height,
            "upscale_method": upscale_method,
            "crop": crop
        }
        return (upscaled,)


@register_comfyui_node("LatentComposite")
class LatentComposite(ComfyUINodeInterface):
    """Composite two latent images"""
    
    CATEGORY = "latent"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples_to": ("LATENT",),
                "samples_from": ("LATENT",),
                "x": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "y": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 8192}),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ("LATENT",)
    
    def execute(self, samples_to, samples_from, x, y, feather):
        composite = {
            "type": "LATENT",
            "base": samples_to,
            "overlay": samples_from,
            "x": x,
            "y": y,
            "feather": feather
        }
        return (composite,)
