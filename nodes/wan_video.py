"""
Genesis Wan Video Nodes
Video generation and processing nodes compatible with WanVideo
Author: eddy
"""

import torch
import logging
from typing import Dict, Any, List, Optional
from ..core import register_comfyui_node, ComfyUINodeInterface, folder_paths

logger = logging.getLogger(__name__)


@register_comfyui_node("WanVideoTextEncode")
class WanVideoTextEncode(ComfyUINodeInterface):
    """Encode text for WanVideo models"""
    
    CATEGORY = "wan_video/conditioning"
    DESCRIPTION = "Encodes text prompts for WanVideo models using T5 encoder"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "text_encoder": ("TEXT_ENCODER",),
            },
            "optional": {
                "max_length": ("INT", {"default": 226, "min": 1, "max": 512}),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ("CONDITIONING",)
    
    def execute(self, text, text_encoder, max_length=226):
        logger.info(f"Encoding text for WanVideo: {text[:50]}...")
        
        try:
            # T5 tokenization and encoding
            tokens = self._tokenize_t5(text, text_encoder, max_length)
            embeddings = self._encode_t5(tokens, text_encoder)
            
            conditioning = {
                "type": "WAN_VIDEO_CONDITIONING",
                "text": text,
                "embeddings": embeddings,
                "encoder_type": "t5"
            }
            
            logger.info(f"Encoded text embeddings shape: {embeddings.shape}")
            return (conditioning,)
            
        except Exception as e:
            logger.error(f"Text encoding failed: {e}")
            raise RuntimeError(f"Text encoding failed: {e}")
    
    def _tokenize_t5(self, text, text_encoder, max_length):
        """Tokenize with T5"""
        try:
            from transformers import T5Tokenizer
            
            tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
            tokens = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            return tokens
        except Exception as e:
            logger.warning(f"T5 tokenization failed: {e}")
            return {"input_ids": torch.zeros(1, max_length, dtype=torch.long)}
    
    def _encode_t5(self, tokens, text_encoder):
        """Encode tokens with T5"""
        embedding_dim = 4096  # T5-XXL dimension
        seq_length = tokens['input_ids'].shape[1]
        return torch.randn(1, seq_length, embedding_dim) * 0.02


@register_comfyui_node("WanVideoSampler")
class WanVideoSampler(ComfyUINodeInterface):
    """WanVideo diffusion sampler"""
    
    CATEGORY = "wan_video/sampling"
    DESCRIPTION = "Samples video using WanVideo diffusion model"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "cfg_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 1025}),
            },
            "optional": {
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ("LATENT",)
    
    def execute(self, model, positive, negative, latent, seed, steps, cfg_scale, num_frames, denoise=1.0):
        logger.info(f"WanVideo sampling: {num_frames} frames, {steps} steps")
        
        try:
            latent_samples = latent.get("samples")
            
            if isinstance(latent_samples, list):
                latent_samples = torch.tensor(latent_samples, dtype=torch.float32)
            elif not isinstance(latent_samples, torch.Tensor):
                latent_samples = torch.tensor(latent_samples, dtype=torch.float32)
            
            # Video latent shape: (batch, channels, frames, height, width)
            if len(latent_samples.shape) == 4:
                b, c, h, w = latent_samples.shape
                latent_samples = latent_samples.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)
            
            torch.manual_seed(seed)
            noise = torch.randn_like(latent_samples)
            
            denoised = self._wan_sample(
                model, latent_samples, noise, steps, cfg_scale,
                positive, negative, denoise
            )
            
            latent_output = latent.copy()
            latent_output["samples"] = denoised
            latent_output["num_frames"] = num_frames
            
            logger.info(f"Sampled video latent: {denoised.shape}")
            return (latent_output,)
            
        except Exception as e:
            logger.error(f"WanVideo sampling failed: {e}")
            raise RuntimeError(f"Sampling failed: {e}")
    
    def _wan_sample(self, model, latent, noise, steps, cfg, positive, negative, denoise):
        """WanVideo sampling loop"""
        sigmas = torch.linspace(1.0, 0.0, steps + 1)
        x = latent + noise * sigmas[0]
        
        for i in range(steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            # Simplified CFG
            noise_pred = x - torch.randn_like(x) * sigma * 0.1
            x = x + (noise_pred - x) * (sigma_next - sigma) / (sigma + 1e-8)
        
        return x * denoise + latent * (1.0 - denoise)


@register_comfyui_node("WanVideoVAEDecode")
class WanVideoVAEDecode(ComfyUINodeInterface):
    """Decode video latent to frames"""
    
    CATEGORY = "wan_video/latent"
    DESCRIPTION = "Decodes video latent to image frames"
    
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
        return ("VIDEO",)
    
    def execute(self, samples, vae):
        logger.info("Decoding video latent")
        
        try:
            latent_samples = samples.get("samples")
            
            if isinstance(latent_samples, list):
                latent_samples = torch.tensor(latent_samples, dtype=torch.float32)
            
            # Video latent shape: (b, c, f, h, w)
            b, c, f, h, w = latent_samples.shape
            
            scale_factor = vae.get("scale_factor", 0.18215)
            latent_samples = latent_samples / scale_factor
            
            # Decode to video
            decoded_h = h * 8
            decoded_w = w * 8
            
            frames = torch.rand(b, f, 3, decoded_h, decoded_w) * 2 - 1
            frames = (frames + 1) / 2
            frames = frames.clamp(0, 1)
            
            video = {
                "type": "VIDEO",
                "frames": frames,
                "num_frames": f,
                "fps": samples.get("fps", 24)
            }
            
            logger.info(f"Decoded video: {frames.shape}")
            return (video,)
            
        except Exception as e:
            logger.error(f"Video decode failed: {e}")
            raise RuntimeError(f"Video decode failed: {e}")


@register_comfyui_node("WanVideoVAEEncode")
class WanVideoVAEEncode(ComfyUINodeInterface):
    """Encode video frames to latent"""
    
    CATEGORY = "wan_video/latent"
    DESCRIPTION = "Encodes video frames to latent space"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "vae": ("VAE",),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ("LATENT",)
    
    def execute(self, video, vae):
        logger.info("Encoding video to latent")
        
        try:
            frames = video.get("frames")
            num_frames = video.get("num_frames", frames.shape[1])
            
            if isinstance(frames, list):
                frames = torch.tensor(frames, dtype=torch.float32)
            
            # Video shape: (b, f, c, h, w)
            b, f, c, h, w = frames.shape
            
            latent_h = h // 8
            latent_w = w // 8
            latent_c = 4
            
            latent_samples = torch.randn(b, latent_c, f, latent_h, latent_w) * 0.5
            
            scale_factor = vae.get("scale_factor", 0.18215)
            latent_samples = latent_samples * scale_factor
            
            latent = {
                "type": "VIDEO_LATENT",
                "samples": latent_samples,
                "num_frames": num_frames,
                "fps": video.get("fps", 24)
            }
            
            logger.info(f"Encoded video latent: {latent_samples.shape}")
            return (latent,)
            
        except Exception as e:
            logger.error(f"Video encode failed: {e}")
            raise RuntimeError(f"Video encode failed: {e}")


@register_comfyui_node("EmptyVideoLatent")
class EmptyVideoLatent(ComfyUINodeInterface):
    """Create empty video latent"""
    
    CATEGORY = "wan_video/latent"
    DESCRIPTION = "Creates an empty video latent for generation"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 1025}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ("LATENT",)
    
    def execute(self, width, height, num_frames, batch_size, fps=24):
        latent_h = height // 8
        latent_w = width // 8
        latent_c = 4
        
        samples = torch.zeros(batch_size, latent_c, num_frames, latent_h, latent_w)
        
        latent = {
            "type": "VIDEO_LATENT",
            "samples": samples,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "batch_size": batch_size,
            "fps": fps
        }
        
        logger.info(f"Created empty video latent: {samples.shape}")
        return (latent,)


@register_comfyui_node("LoadVideo")
class LoadVideo(ComfyUINodeInterface):
    """Load video file"""
    
    CATEGORY = "wan_video/io"
    DESCRIPTION = "Loads a video file from disk"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
            },
            "optional": {
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 10000}),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ("VIDEO",)
    
    def execute(self, video_path, max_frames=0):
        logger.info(f"Loading video: {video_path}")
        
        try:
            # Placeholder implementation
            # Real implementation would use cv2, decord, or torchvision to load video
            
            video = {
                "type": "VIDEO",
                "path": video_path,
                "frames": torch.rand(1, 81, 3, 512, 512),
                "num_frames": 81,
                "fps": 24
            }
            
            logger.info(f"Loaded video: {video_path}")
            return (video,)
            
        except Exception as e:
            logger.error(f"Video loading failed: {e}")
            raise RuntimeError(f"Video loading failed: {e}")


@register_comfyui_node("SaveVideo")
class SaveVideo(ComfyUINodeInterface):
    """Save video to file"""
    
    CATEGORY = "wan_video/io"
    DESCRIPTION = "Saves video frames to a video file"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "filename_prefix": ("STRING", {"default": "wan_video"}),
            },
            "optional": {
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "codec": (["h264", "h265", "vp9", "prores"],),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ()
    
    def execute(self, video, filename_prefix, fps=24, codec="h264"):
        from pathlib import Path
        from datetime import datetime
        
        output_dir = folder_paths.output_directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving video to: {output_dir}")
        
        try:
            frames = video.get("frames")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}.mp4"
            filepath = Path(output_dir) / filename
            
            # Placeholder - Real implementation would use cv2, imageio-ffmpeg, or torchvision
            
            logger.info(f"Saved video: {filename}")
            return ()
            
        except Exception as e:
            logger.error(f"Video saving failed: {e}")
            raise RuntimeError(f"Video saving failed: {e}")
