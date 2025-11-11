"""
Genesis VAE
Variational Autoencoder for latent encoding/decoding
Author: eddy
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging


class VAE:
    """
    VAE wrapper for encoding and decoding
    
    Handles conversion between pixel space and latent space
    """
    
    def __init__(self, model: Optional[nn.Module] = None, device: str = "cuda"):
        """
        Initialize VAE
        
        Args:
            model: VAE model (loaded from checkpoint)
            device: Computing device
        """
        self.model = model
        self.device = torch.device(device)
        self.logger = logging.getLogger('Genesis.VAE')
        
        # VAE scaling factor (SD 1.5/2.x uses 0.18215, SDXL uses different)
        self.scale_factor = 0.18215
        
        if self.model:
            self.model.to(self.device)
            self.model.eval()
    
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent space
        
        Args:
            images: Image tensor [B, C, H, W] in range [-1, 1]
            
        Returns:
            Latent tensor [B, 4, H//8, W//8]
        """
        if self.model is None:
            raise RuntimeError("VAE model not loaded")
        
        with torch.no_grad():
            images = images.to(self.device)
            
            # Encode
            latent_dist = self.model.encode(images)
            
            # Sample from distribution (use mean for deterministic)
            if hasattr(latent_dist, 'sample'):
                latent = latent_dist.sample()
            elif hasattr(latent_dist, 'latent_dist'):
                latent = latent_dist.latent_dist.sample()
            else:
                latent = latent_dist
            
            # Scale
            latent = latent * self.scale_factor
            
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to images
        
        Args:
            latent: Latent tensor [B, 4, H//8, W//8]
            
        Returns:
            Image tensor [B, C, H, W] in range [-1, 1]
        """
        if self.model is None:
            raise RuntimeError("VAE model not loaded")
        
        with torch.no_grad():
            latent = latent.to(self.device)
            
            # Unscale
            latent = latent / self.scale_factor
            
            # Decode
            images = self.model.decode(latent)
            
            if hasattr(images, 'sample'):
                images = images.sample
            
        return images
    
    def encode_tiled(
        self,
        images: torch.Tensor,
        tile_size: int = 512,
        overlap: int = 64
    ) -> torch.Tensor:
        """
        Encode images using tiling (for large images)
        
        Args:
            images: Image tensor
            tile_size: Tile size in pixels
            overlap: Overlap between tiles
            
        Returns:
            Latent tensor
        """
        # TODO: Implement tiled encoding for memory efficiency
        return self.encode(images)
    
    def decode_tiled(
        self,
        latent: torch.Tensor,
        tile_size: int = 64,
        overlap: int = 8
    ) -> torch.Tensor:
        """
        Decode latent using tiling (for large images)
        
        Args:
            latent: Latent tensor
            tile_size: Tile size in latent space
            overlap: Overlap between tiles
            
        Returns:
            Image tensor
        """
        # TODO: Implement tiled decoding for memory efficiency
        return self.decode(latent)
    
    def get_latent_size(self, image_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate latent size from image size
        
        Args:
            image_size: (height, width) in pixels
            
        Returns:
            (latent_height, latent_width)
        """
        h, w = image_size
        return (h // 8, w // 8)
    
    def get_image_size(self, latent_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate image size from latent size
        
        Args:
            latent_size: (latent_height, latent_width)
            
        Returns:
            (height, width) in pixels
        """
        h, w = latent_size
        return (h * 8, w * 8)
    
    def create_empty_latent(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: Optional[str] = None
    ) -> torch.Tensor:
        """
        Create empty latent tensor
        
        Args:
            batch_size: Batch size
            height: Image height in pixels
            width: Image width in pixels
            device: Device to create tensor on
            
        Returns:
            Empty latent tensor filled with zeros
        """
        device = device or self.device
        latent_h, latent_w = self.get_latent_size((height, width))
        
        latent = torch.zeros(
            batch_size,
            4,  # Latent channels
            latent_h,
            latent_w,
            device=device
        )
        
        return latent
    
    def create_random_latent(
        self,
        batch_size: int,
        height: int,
        width: int,
        seed: Optional[int] = None,
        device: Optional[str] = None
    ) -> torch.Tensor:
        """
        Create random latent tensor
        
        Args:
            batch_size: Batch size
            height: Image height in pixels
            width: Image width in pixels
            seed: Random seed
            device: Device to create tensor on
            
        Returns:
            Random latent tensor
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        device = device or self.device
        latent_h, latent_w = self.get_latent_size((height, width))
        
        latent = torch.randn(
            batch_size,
            4,
            latent_h,
            latent_w,
            device=device
        )
        
        return latent


class VAEProcessor:
    """Helper class for VAE image preprocessing and postprocessing"""
    
    @staticmethod
    def preprocess_image(image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image for VAE encoding
        
        Args:
            image: Image tensor in range [0, 1]
            
        Returns:
            Preprocessed image in range [-1, 1]
        """
        return image * 2.0 - 1.0
    
    @staticmethod
    def postprocess_image(image: torch.Tensor) -> torch.Tensor:
        """
        Postprocess image from VAE decoding
        
        Args:
            image: Image tensor in range [-1, 1]
            
        Returns:
            Postprocessed image in range [0, 1]
        """
        image = (image + 1.0) / 2.0
        return torch.clamp(image, 0.0, 1.0)
    
    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor):
        """
        Convert tensor to PIL Image
        
        Args:
            tensor: Image tensor [C, H, W] or [B, C, H, W]
            
        Returns:
            PIL Image or list of PIL Images
        """
        from PIL import Image
        import numpy as np
        
        # Handle batch
        if tensor.dim() == 4:
            images = []
            for i in range(tensor.shape[0]):
                img = VAEProcessor.tensor_to_pil(tensor[i])
                images.append(img)
            return images
        
        # Single image [C, H, W]
        tensor = tensor.cpu().detach()
        tensor = torch.clamp(tensor, 0.0, 1.0)
        
        # Convert to numpy
        array = tensor.permute(1, 2, 0).numpy()
        array = (array * 255).astype(np.uint8)
        
        return Image.fromarray(array)
    
    @staticmethod
    def pil_to_tensor(image, device: str = "cuda") -> torch.Tensor:
        """
        Convert PIL Image to tensor
        
        Args:
            image: PIL Image or list of PIL Images
            device: Target device
            
        Returns:
            Image tensor [B, C, H, W] in range [0, 1]
        """
        import numpy as np
        from PIL import Image
        
        # Handle list
        if isinstance(image, list):
            tensors = [VAEProcessor.pil_to_tensor(img, device) for img in image]
            return torch.cat(tensors, dim=0)
        
        # Single image
        if not isinstance(image, Image.Image):
            raise TypeError("Expected PIL Image")
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # To numpy
        array = np.array(image).astype(np.float32) / 255.0
        
        # To tensor [H, W, C] -> [C, H, W]
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(device)
