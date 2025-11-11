"""
Genesis Model Loader
Model loader - Extracted and simplified from ComfyUI core
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging


class ModelLoader:
    """
    Model Loader
    
    Responsible for loading and managing Stable Diffusion models
    """
    
    def __init__(self, config, device):
        """
        Initialize model loader
        
        Args:
            config: Genesis configuration
            device: Computing device
        """
        self.config = config
        self.device = device
        self.logger = logging.getLogger('Genesis.ModelLoader')
        
        # Loaded models cache
        self.loaded_models = {}
        
    def list_checkpoints(self) -> List[str]:
        """
        List available checkpoint models
        
        Returns:
            Checkpoint file list
        """
        checkpoint_dir = self.config.checkpoints_dir
        if not checkpoint_dir.exists():
            self.logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
            return []
        
        extensions = ['.safetensors', '.ckpt', '.pt', '.pth', '.bin']
        checkpoints = []
        
        for ext in extensions:
            checkpoints.extend([
                f.name for f in checkpoint_dir.glob(f'*{ext}')
            ])
        
        return sorted(checkpoints)
    
    def list_vae(self) -> List[str]:
        """
        List available VAE models
        
        Returns:
            VAE file list
        """
        vae_dir = self.config.vae_dir
        if not vae_dir.exists():
            return []
        
        extensions = ['.safetensors', '.pt', '.pth', '.bin']
        vae_files = []
        
        for ext in extensions:
            vae_files.extend([
                f.name for f in vae_dir.glob(f'*{ext}')
            ])
        
        return sorted(vae_files)
    
    def list_loras(self) -> List[str]:
        """
        List available LoRA models
        
        Returns:
            LoRA file list
        """
        lora_dir = self.config.lora_dir
        if not lora_dir.exists():
            return []
        
        extensions = ['.safetensors', '.pt', '.pth', '.bin']
        lora_files = []
        
        for ext in extensions:
            lora_files.extend([
                f.name for f in lora_dir.glob(f'*{ext}')
            ])
        
        return sorted(lora_files)
    
    def load_checkpoint(self, checkpoint_name: str) -> Dict[str, Any]:
        """
        Load checkpoint model
        
        Args:
            checkpoint_name: Checkpoint filename
            
        Returns:
            Model information dictionary
        """
        # Check cache
        if checkpoint_name in self.loaded_models:
            self.logger.info(f"Using cached model: {checkpoint_name}")
            return self.loaded_models[checkpoint_name]
        
        checkpoint_path = self.config.checkpoints_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.logger.info(f"Loading checkpoint: {checkpoint_name}")
        
        # TODO: Actual model loading logic
        # Need to extract loading logic from ComfyUI's comfy/sd.py
        
        model_info = {
            'name': checkpoint_name,
            'path': str(checkpoint_path),
            'type': 'checkpoint',
            'loaded': True,
            # 'model': actual_model,
            # 'clip': clip_model,
            # 'vae': vae_model,
        }
        
        # Cache model
        self.loaded_models[checkpoint_name] = model_info
        
        self.logger.info(f"✓ Checkpoint loaded: {checkpoint_name}")
        return model_info
    
    def load_vae(self, vae_name: str) -> Dict[str, Any]:
        """
        Load VAE model
        
        Args:
            vae_name: VAE filename
            
        Returns:
            VAE model information
        """
        vae_path = self.config.vae_dir / vae_name
        
        if not vae_path.exists():
            raise FileNotFoundError(f"VAE not found: {vae_path}")
        
        self.logger.info(f"Loading VAE: {vae_name}")
        
        # TODO: Actual VAE loading logic
        
        vae_info = {
            'name': vae_name,
            'path': str(vae_path),
            'type': 'vae',
            # 'model': actual_vae,
        }
        
        return vae_info
    
    def load_lora(self, lora_name: str, strength: float = 1.0) -> Dict[str, Any]:
        """
        Load LoRA model
        
        Args:
            lora_name: LoRA filename
            strength: LoRA strength
            
        Returns:
            LoRA model information
        """
        lora_path = self.config.lora_dir / lora_name
        
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA not found: {lora_path}")
        
        self.logger.info(f"Loading LoRA: {lora_name} (strength: {strength})")
        
        # TODO: Actual LoRA loading logic
        
        lora_info = {
            'name': lora_name,
            'path': str(lora_path),
            'type': 'lora',
            'strength': strength,
            # 'model': actual_lora,
        }
        
        return lora_info
    
    def unload_model(self, model_name: str):
        """
        Unload model
        
        Args:
            model_name: Model name
        """
        if model_name in self.loaded_models:
            self.logger.info(f"Unloading model: {model_name}")
            del self.loaded_models[model_name]
            
            # Clear GPU memory
            if self.device.type == 'cuda':
                import torch
                torch.cuda.empty_cache()
    
    def cleanup(self):
        """Cleanup all loaded models"""
        self.logger.info("Cleaning up all loaded models...")
        self.loaded_models.clear()
        
        if self.device.type == 'cuda':
            import torch
            torch.cuda.empty_cache()
