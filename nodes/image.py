"""
Genesis Image Nodes
Image processing and I/O nodes
Author: eddy
"""

import os
import torch
import numpy as np
import logging
from PIL import Image
from pathlib import Path
from datetime import datetime
from ..core import register_comfyui_node, ComfyUINodeInterface, folder_paths

logger = logging.getLogger(__name__)


@register_comfyui_node("LoadImage")
class LoadImage(ComfyUINodeInterface):
    """Load image from file"""
    
    CATEGORY = "image"
    DESCRIPTION = "Loads an image file from the input directory"
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_folder_paths("input")
        files = []
        if input_dir:
            for dir_path in input_dir:
                if os.path.exists(dir_path):
                    files.extend([f for f in os.listdir(dir_path) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))])
        
        return {
            "required": {
                "image": (sorted(files) if files else ["no_images.png"],),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ("IMAGE", "MASK")
    
    def execute(self, image):
        input_dir = folder_paths.get_folder_paths("input")
        image_path = None
        if input_dir:
            for dir_path in input_dir:
                potential_path = os.path.join(dir_path, image)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
        
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image}")
        
        logger.info(f"Loading image: {image}")
        
        try:
            img = Image.open(image_path)
            
            if img.mode == 'I':
                img = img.point(lambda i: i * (1 / 256)).convert('L')
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_array = np.array(img).astype(np.float32) / 255.0
            
            pixels = torch.from_numpy(img_array)
            if len(pixels.shape) == 2:
                pixels = pixels.unsqueeze(-1).repeat(1, 1, 3)
            
            pixels = pixels.unsqueeze(0)
            pixels = pixels.permute(0, 3, 1, 2)
            
            mask = torch.ones((1, 1, pixels.shape[2], pixels.shape[3]))
            
            if img.mode == 'RGBA':
                alpha = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                mask = torch.from_numpy(alpha).unsqueeze(0).unsqueeze(0)
            
            image_data = {
                "type": "IMAGE",
                "path": image_path,
                "filename": image,
                "pixels": pixels
            }
            
            mask_data = {
                "type": "MASK",
                "data": mask
            }
            
            logger.info(f"Loaded image: {pixels.shape}")
            return (image_data, mask_data)
            
        except Exception as e:
            logger.error(f"Failed to load image {image}: {e}")
            raise RuntimeError(f"Failed to load image: {e}")


@register_comfyui_node("SaveImage")
class SaveImage(ComfyUINodeInterface):
    """Save image to file"""
    
    CATEGORY = "image"
    DESCRIPTION = "Saves generated images to the output directory"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "genesis"}),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ()
    
    def execute(self, images, filename_prefix):
        output_dir = folder_paths.output_directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving images to: {output_dir}")
        
        try:
            pixels = images.get("pixels")
            
            if pixels is None:
                raise ValueError("No pixel data in image")
            
            if isinstance(pixels, list):
                pixels = torch.tensor(pixels, dtype=torch.float32)
            elif not isinstance(pixels, torch.Tensor):
                pixels = torch.tensor(pixels, dtype=torch.float32)
            
            if len(pixels.shape) == 3:
                pixels = pixels.unsqueeze(0)
            
            if pixels.shape[1] == 3 or pixels.shape[1] == 4:
                pixels = pixels.permute(0, 2, 3, 1)
            
            pixels = pixels.cpu().numpy()
            pixels = (pixels * 255).clip(0, 255).astype(np.uint8)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            saved_files = []
            for i, img_array in enumerate(pixels):
                filename = f"{filename_prefix}_{timestamp}_{i:04d}.png"
                filepath = os.path.join(output_dir, filename)
                
                img = Image.fromarray(img_array)
                img.save(filepath, format='PNG')
                
                saved_files.append(filepath)
                logger.info(f"Saved image: {filename}")
            
            logger.info(f"Successfully saved {len(saved_files)} images")
            return ()
            
        except Exception as e:
            logger.error(f"Failed to save images: {e}")
            raise RuntimeError(f"Failed to save images: {e}")


@register_comfyui_node("PreviewImage")
class PreviewImage(ComfyUINodeInterface):
    """Preview image"""
    
    CATEGORY = "image"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ()
    
    def execute(self, images):
        return ()


@register_comfyui_node("ImageScale")
class ImageScale(ComfyUINodeInterface):
    """Scale image"""
    
    CATEGORY = "image"
    DESCRIPTION = "Resizes images to specified dimensions"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "crop": (["disabled", "center"],),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ("IMAGE",)
    
    def execute(self, image, upscale_method, width, height, crop):
        logger.info(f"Scaling image to {width}x{height} using {upscale_method}")
        
        try:
            pixels = image.get("pixels") if isinstance(image, dict) else image
            
            if isinstance(pixels, list):
                pixels = torch.tensor(pixels, dtype=torch.float32)
            elif not isinstance(pixels, torch.Tensor):
                pixels = torch.tensor(pixels, dtype=torch.float32)
            
            if len(pixels.shape) == 3:
                pixels = pixels.unsqueeze(0)
            
            if pixels.shape[1] != 3 and pixels.shape[1] != 4:
                pixels = pixels.permute(0, 3, 1, 2)
            
            import torch.nn.functional as F
            
            mode_map = {
                "nearest-exact": "nearest",
                "bilinear": "bilinear",
                "area": "area",
                "bicubic": "bicubic",
                "lanczos": "bilinear"
            }
            
            mode = mode_map.get(upscale_method, "bilinear")
            
            scaled = F.interpolate(
                pixels,
                size=(height, width),
                mode=mode,
                align_corners=False if mode != "nearest" else None
            )
            
            if crop == "center" and (scaled.shape[2] != height or scaled.shape[3] != width):
                h, w = scaled.shape[2], scaled.shape[3]
                top = (h - height) // 2
                left = (w - width) // 2
                scaled = scaled[:, :, top:top+height, left:left+width]
            
            scaled_image = image.copy() if isinstance(image, dict) else {}
            scaled_image["pixels"] = scaled
            scaled_image["width"] = width
            scaled_image["height"] = height
            
            logger.info(f"Scaled image shape: {scaled.shape}")
            return (scaled_image,)
            
        except Exception as e:
            logger.error(f"Image scaling failed: {e}")
            raise RuntimeError(f"Image scaling failed: {e}")


@register_comfyui_node("ImageBatch")
class ImageBatch(ComfyUINodeInterface):
    """Batch images"""
    
    CATEGORY = "image"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ("IMAGE",)
    
    def execute(self, image1, image2):
        batched = {
            "type": "IMAGE",
            "batch": [image1, image2]
        }
        return (batched,)


@register_comfyui_node("ImageInvert")
class ImageInvert(ComfyUINodeInterface):
    """Invert image colors"""
    
    CATEGORY = "image"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ("IMAGE",)
    
    def execute(self, image):
        inverted = {
            **image,
            "inverted": True
        }
        return (inverted,)
