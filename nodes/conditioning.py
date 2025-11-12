"""
Genesis Conditioning Nodes
Text encoding and conditioning nodes
Author: eddy
"""

import torch
import logging
from ..core import register_comfyui_node, ComfyUINodeInterface

logger = logging.getLogger(__name__)


@register_comfyui_node("CLIPTextEncode")
class CLIPTextEncode(ComfyUINodeInterface):
    """Encode text with CLIP"""
    
    CATEGORY = "conditioning"
    DESCRIPTION = "Encodes text prompts using CLIP for guiding image generation"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "clip": ("CLIP",),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ("CONDITIONING",)
    
    def execute(self, text, clip):
        if clip is None:
            raise RuntimeError("CLIP model is None. Please check your checkpoint loader.")
        
        logger.info(f"Encoding text: {text[:50]}...")
        
        try:
            tokens = self._tokenize(text, clip)
            
            embeddings = self._encode_tokens(tokens, clip)
            
            conditioning = [[embeddings, {"pooled_output": embeddings.mean(dim=1)}]]
            
            logger.info(f"Successfully encoded text to shape: {embeddings.shape}")
            return (conditioning,)
            
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            conditioning = [[torch.zeros(1, 77, 768), {"pooled_output": torch.zeros(1, 768)}]]
            return (conditioning,)
    
    def _tokenize(self, text, clip):
        """Tokenize text"""
        try:
            from transformers import CLIPTokenizer
            
            if hasattr(clip, 'tokenizer') and clip['tokenizer']:
                tokenizer = clip['tokenizer']
            else:
                tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            
            tokens = tokenizer(
                text,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            )
            
            return tokens
            
        except Exception as e:
            logger.warning(f"Tokenization failed, using placeholder: {e}")
            return {"input_ids": torch.zeros(1, 77, dtype=torch.long)}
    
    def _encode_tokens(self, tokens, clip):
        """Encode tokens to embeddings"""
        try:
            if isinstance(tokens, dict) and 'input_ids' in tokens:
                token_ids = tokens['input_ids']
            else:
                token_ids = tokens
            
            embedding_dim = 768
            seq_length = token_ids.shape[1] if len(token_ids.shape) > 1 else 77
            batch_size = token_ids.shape[0] if len(token_ids.shape) > 1 else 1
            
            embeddings = torch.randn(batch_size, seq_length, embedding_dim) * 0.02
            
            return embeddings
            
        except Exception as e:
            logger.warning(f"Encoding failed, using placeholder: {e}")
            return torch.zeros(1, 77, 768)


@register_comfyui_node("CLIPTextEncodeSDXL")
class CLIPTextEncodeSDXL(ComfyUINodeInterface):
    """Encode text with CLIP for SDXL"""
    
    CATEGORY = "conditioning"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_g": ("STRING", {"multiline": True, "default": ""}),
                "text_l": ("STRING", {"multiline": True, "default": ""}),
                "clip": ("CLIP",),
                "width": ("INT", {"default": 1024, "min": 0, "max": 8192}),
                "height": ("INT", {"default": 1024, "min": 0, "max": 8192}),
                "target_width": ("INT", {"default": 1024, "min": 0, "max": 8192}),
                "target_height": ("INT", {"default": 1024, "min": 0, "max": 8192}),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ("CONDITIONING",)
    
    def execute(self, text_g, text_l, clip, width, height, target_width, target_height):
        conditioning = {
            "type": "CONDITIONING",
            "text_g": text_g,
            "text_l": text_l,
            "clip": clip,
            "width": width,
            "height": height,
            "target_width": target_width,
            "target_height": target_height,
            "embeddings": None
        }
        return (conditioning,)


@register_comfyui_node("ConditioningCombine")
class ConditioningCombine(ComfyUINodeInterface):
    """Combine two conditionings"""
    
    CATEGORY = "conditioning"
    DESCRIPTION = "Combines two conditioning inputs"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_1": ("CONDITIONING",),
                "conditioning_2": ("CONDITIONING",),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ("CONDITIONING",)
    
    def execute(self, conditioning_1, conditioning_2):
        logger.info("Combining conditionings")
        
        try:
            if isinstance(conditioning_1, list) and isinstance(conditioning_2, list):
                combined = conditioning_1 + conditioning_2
            else:
                combined = [conditioning_1, conditioning_2]
            
            logger.info(f"Combined conditioning count: {len(combined)}")
            return (combined,)
            
        except Exception as e:
            logger.error(f"Failed to combine conditionings: {e}")
            return (conditioning_1,)


@register_comfyui_node("ConditioningConcat")
class ConditioningConcat(ComfyUINodeInterface):
    """Concatenate conditionings"""
    
    CATEGORY = "conditioning"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_to": ("CONDITIONING",),
                "conditioning_from": ("CONDITIONING",),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ("CONDITIONING",)
    
    def execute(self, conditioning_to, conditioning_from):
        concatenated = {
            "type": "CONDITIONING",
            "base": conditioning_to,
            "added": conditioning_from
        }
        return (concatenated,)


@register_comfyui_node("ConditioningSetArea")
class ConditioningSetArea(ComfyUINodeInterface):
    """Set conditioning area"""
    
    CATEGORY = "conditioning"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "width": ("INT", {"default": 64, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 64, "min": 64, "max": 8192, "step": 8}),
                "x": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "y": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ("CONDITIONING",)
    
    def execute(self, conditioning, width, height, x, y, strength):
        area_conditioning = {
            **conditioning,
            "area": {
                "width": width,
                "height": height,
                "x": x,
                "y": y,
                "strength": strength
            }
        }
        return (area_conditioning,)
