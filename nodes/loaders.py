"""
Genesis Loader Nodes
Model loading nodes compatible with ComfyUI
Author: eddy
"""

import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import folder_paths (for ComfyUI compatibility)
try:
    from ..core import folder_paths
except ImportError:
    # Fallback if not available
    class folder_paths:
        @staticmethod
        def get_filename_list(folder_type):
            return []

        @staticmethod
        def get_full_path(folder_type, filename):
            return filename


class CheckpointLoaderSimple:
    """Load checkpoint model"""

    CATEGORY = "loaders"
    DESCRIPTION = "Loads a diffusion model checkpoint for image generation"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_checkpoint"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            }
        }

    def load_checkpoint(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        logger.info(f"Loading checkpoint: {ckpt_name}")
        
        try:
            import safetensors.torch
            
            if ckpt_path.endswith('.safetensors'):
                state_dict = safetensors.torch.load_file(ckpt_path, device="cpu")
            else:
                state_dict = torch.load(ckpt_path, map_location="cpu")
            
            model = {
                "type": "MODEL",
                "path": ckpt_path,
                "name": ckpt_name,
                "state_dict": state_dict,
                "config": self._detect_model_config(state_dict)
            }
            
            clip = {
                "type": "CLIP",
                "path": ckpt_path,
                "state_dict": state_dict,
                "tokenizer": None
            }
            
            vae = {
                "type": "VAE",
                "path": ckpt_path,
                "state_dict": state_dict
            }
            
            logger.info(f"Successfully loaded checkpoint: {ckpt_name}")
            return (model, clip, vae)
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {ckpt_name}: {e}")
            raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    def _detect_model_config(self, state_dict):
        """Detect model configuration from state dict"""
        config = {
            "model_type": "unknown",
            "base_model": "sd15"
        }
        
        if state_dict:
            keys = list(state_dict.keys())
            
            if any("conditioner" in k for k in keys):
                config["base_model"] = "sdxl"
            elif any("cond_stage_model" in k for k in keys):
                config["base_model"] = "sd15"
        
        return config


class VAELoader:
    """Load VAE model"""

    CATEGORY = "loaders"
    DESCRIPTION = "Loads a VAE model for encoding/decoding images to/from latent space"
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"),),
            }
        }

    def load_vae(self, vae_name):
        vae_path = folder_paths.get_full_path("vae", vae_name)
        
        if not Path(vae_path).exists():
            raise FileNotFoundError(f"VAE not found: {vae_path}")
        
        logger.info(f"Loading VAE: {vae_name}")
        
        try:
            import safetensors.torch
            
            if vae_path.endswith('.safetensors'):
                state_dict = safetensors.torch.load_file(vae_path, device="cpu")
            else:
                state_dict = torch.load(vae_path, map_location="cpu")
            
            vae = {
                "type": "VAE",
                "path": vae_path,
                "name": vae_name,
                "state_dict": state_dict,
                "scale_factor": 0.18215
            }
            
            logger.info(f"Successfully loaded VAE: {vae_name}")
            return (vae,)
            
        except Exception as e:
            logger.error(f"Failed to load VAE {vae_name}: {e}")
            raise RuntimeError(f"Failed to load VAE: {e}")


class LoraLoader:
    """Load LoRA model"""

    CATEGORY = "loaders"
    DESCRIPTION = "Applies LoRA adaptations to model and CLIP"
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "load_lora"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        
        if not Path(lora_path).exists():
            raise FileNotFoundError(f"LoRA not found: {lora_path}")
        
        logger.info(f"Loading LoRA: {lora_name} (model:{strength_model}, clip:{strength_clip})")
        
        try:
            import safetensors.torch
            
            if lora_path.endswith('.safetensors'):
                lora_state_dict = safetensors.torch.load_file(lora_path, device="cpu")
            else:
                lora_state_dict = torch.load(lora_path, map_location="cpu")
            
            model_with_lora = model.copy() if isinstance(model, dict) else model
            clip_with_lora = clip.copy() if isinstance(clip, dict) else clip
            
            if isinstance(model_with_lora, dict):
                if "loras" not in model_with_lora:
                    model_with_lora["loras"] = []
                model_with_lora["loras"].append({
                    "path": lora_path,
                    "name": lora_name,
                    "state_dict": lora_state_dict,
                    "strength": strength_model
                })
            
            if isinstance(clip_with_lora, dict):
                if "loras" not in clip_with_lora:
                    clip_with_lora["loras"] = []
                clip_with_lora["loras"].append({
                    "path": lora_path,
                    "name": lora_name,
                    "state_dict": lora_state_dict,
                    "strength": strength_clip
                })
            
            logger.info(f"Successfully loaded LoRA: {lora_name}")
            return (model_with_lora, clip_with_lora)
            
        except Exception as e:
            logger.error(f"Failed to load LoRA {lora_name}: {e}")
            raise RuntimeError(f"Failed to load LoRA: {e}")


class CLIPLoader:
    """Load CLIP model"""

    CATEGORY = "loaders"
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_clip"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("clip"),),
            }
        }

    def load_clip(self, clip_name):
        clip_path = folder_paths.get_full_path("clip", clip_name)
        clip = {"type": "CLIP", "path": clip_path, "name": clip_name}
        return (clip,)


class ControlNetLoader:
    """Load ControlNet model"""

    CATEGORY = "loaders"
    RETURN_TYPES = ("CONTROL_NET",)
    RETURN_NAMES = ("control_net",)
    FUNCTION = "load_controlnet"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_net_name": (folder_paths.get_filename_list("controlnet"),),
            }
        }

    def load_controlnet(self, control_net_name):
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet = {"type": "CONTROL_NET", "path": controlnet_path, "name": control_net_name}
        return (controlnet,)


# ComfyUI-compatible NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "CheckpointLoaderSimple": CheckpointLoaderSimple,
    "VAELoader": VAELoader,
    "LoraLoader": LoraLoader,
    "CLIPLoader": CLIPLoader,
    "ControlNetLoader": ControlNetLoader,
}

# Display names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "CheckpointLoaderSimple": "Load Checkpoint",
    "VAELoader": "Load VAE",
    "LoraLoader": "Load LoRA",
    "CLIPLoader": "Load CLIP",
    "ControlNetLoader": "Load ControlNet",
}
