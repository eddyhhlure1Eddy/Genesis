"""
Genesis Model Loader
Model loader - Extracted and simplified from ComfyUI core
Author: eddy
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import torch
import safetensors.torch


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
        if checkpoint_name in self.loaded_models:
            self.logger.info(f"Using cached model: {checkpoint_name}")
            return self.loaded_models[checkpoint_name]

        checkpoint_path = self.config.checkpoints_dir / checkpoint_name

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.logger.info(f"Loading checkpoint: {checkpoint_name}")

        state_dict = self._load_state_dict(checkpoint_path)

        model_info = {
            'name': checkpoint_name,
            'path': str(checkpoint_path),
            'type': 'checkpoint',
            'loaded': True,
            'state_dict': state_dict,
            'keys_count': len(state_dict.keys()),
        }

        self.loaded_models[checkpoint_name] = model_info

        self.logger.info(f"✓ Checkpoint loaded: {checkpoint_name} ({len(state_dict)} keys)")
        return model_info

    def _load_state_dict(self, path: Path) -> Dict[str, torch.Tensor]:
        """
        Load state dict from file

        Args:
            path: Model file path

        Returns:
            State dictionary
        """
        suffix = path.suffix.lower()

        try:
            if suffix == '.safetensors':
                self.logger.debug(f"Loading safetensors: {path.name}")
                state_dict = safetensors.torch.load_file(str(path), device='cpu')
            elif suffix in ['.ckpt', '.pt', '.pth', '.bin']:
                self.logger.debug(f"Loading checkpoint: {path.name}")
                checkpoint = torch.load(str(path), map_location='cpu', weights_only=True)

                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
            else:
                raise ValueError(f"Unsupported file format: {suffix}")

            return state_dict

        except Exception as e:
            self.logger.error(f"Failed to load {path.name}: {e}")
            raise
    
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

        state_dict = self._load_state_dict(vae_path)

        vae_info = {
            'name': vae_name,
            'path': str(vae_path),
            'type': 'vae',
            'state_dict': state_dict,
            'keys_count': len(state_dict.keys()),
        }

        self.logger.info(f"✓ VAE loaded: {vae_name} ({len(state_dict)} keys)")
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

        state_dict = self._load_state_dict(lora_path)

        lora_info = {
            'name': lora_name,
            'path': str(lora_path),
            'type': 'lora',
            'strength': strength,
            'state_dict': state_dict,
            'keys_count': len(state_dict.keys()),
        }

        self.logger.info(f"✓ LoRA loaded: {lora_name} (strength: {strength}, {len(state_dict)} keys)")
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
