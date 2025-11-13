"""
Genesis Engine - Core Engine Implementation
Author: eddy
Date: 2025-11-12
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch

from .config import GenesisConfig
from ..models.loader import ModelLoader
from ..execution.executor import Executor
from ..utils.logger import setup_logger


class GenesisEngine:
    """
    Genesis Engine - Simple yet powerful AI image generation engine
    
    Example:
        >>> from genesis import GenesisEngine
        >>> engine = GenesisEngine()
        >>> engine.initialize()
        >>> result = engine.generate(prompt="a beautiful landscape")
    """
    
    def __init__(self, config: Optional[GenesisConfig] = None):
        """
        Initialize Genesis Engine
        
        Args:
            config: Engine configuration, uses default if None
        """
        self.config = config or GenesisConfig()
        self.logger = setup_logger('Genesis', self.config.log_level)
        
        # Components
        self.model_loader: Optional[ModelLoader] = None
        self.executor: Optional[Executor] = None
        self.optimizer: Optional[Any] = None

        # State
        self._initialized = False
        self._device = None

        self.logger.info(f"Genesis Engine v0.1.0 initialized")
        
    def initialize(self):
        """
        Initialize the engine
        
        This method will:
        1. Create necessary directories
        2. Detect and configure device (GPU/CPU)
        3. Initialize model loader
        4. Initialize executor
        """
        if self._initialized:
            self.logger.warning("Engine already initialized")
            return
        
        self.logger.info("Initializing Genesis Engine...")
        
        # Create directories
        self.config.create_directories()
        self.logger.debug("Directories created")
        
        # Detect device
        self._setup_device()

        # Apply core optimizations
        self._apply_optimizations()

        # Initialize components
        self.model_loader = ModelLoader(self.config, self._device)
        self.executor = Executor(self.config, self._device)

        self._initialized = True
        self.logger.info("[OK] Genesis Engine initialized successfully")
        
    def _setup_device(self):
        """Setup computing device"""
        if self.config.device == 'cuda':
            if torch.cuda.is_available():
                self._device = torch.device(f'cuda:{self.config.device_id}')
                gpu_name = torch.cuda.get_device_name(self.config.device_id)
                vram_gb = torch.cuda.get_device_properties(self.config.device_id).total_memory / (1024**3)
                self.logger.info(f"[OK] Using GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
            else:
                self.logger.warning("CUDA not available, falling back to CPU")
                self._device = torch.device('cpu')
        elif self.config.device == 'mps':
            if torch.backends.mps.is_available():
                self._device = torch.device('mps')
                self.logger.info("[OK] Using Apple Metal (MPS)")
            else:
                self.logger.warning("MPS not available, falling back to CPU")
                self._device = torch.device('cpu')
        else:
            self._device = torch.device('cpu')
            self.logger.info("[OK] Using CPU")
            
    def _apply_optimizations(self):
        """Apply core PyTorch and CUDA optimizations"""
        try:
            from .optimization import CoreOptimizer

            self.optimizer = CoreOptimizer(self._device)
            results = self.optimizer.apply_all_optimizations(
                enable_tf32=getattr(self.config, 'allow_tf32', True),
                enable_cudnn_benchmark=True,
                enable_jit_fusion=True
            )

            applied = sum(1 for v in results.values() if v)
            self.logger.info(f"[OK] Applied {applied}/{len(results)} core optimizations")

            if self.config.log_level == logging.DEBUG:
                self.optimizer.print_optimization_report()

        except ImportError:
            self.logger.warning("optimization module not found")
        except Exception as e:
            self.logger.warning(f"Could not apply optimizations: {e}")
    
    def load_model(self, checkpoint_name: str, vae_name: Optional[str] = None) -> Dict:
        """
        Load model
        
        Args:
            checkpoint_name: Checkpoint filename
            vae_name: VAE filename (optional)
            
        Returns:
            Loaded model information
        """
        self._ensure_initialized()
        
        self.logger.info(f"Loading model: {checkpoint_name}")
        
        model_info = self.model_loader.load_checkpoint(checkpoint_name)
        
        if vae_name:
            self.logger.info(f"Loading VAE: {vae_name}")
            vae = self.model_loader.load_vae(vae_name)
            model_info['vae'] = vae
            
        return model_info
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate image (simplified interface)
        
        Args:
            prompt: Positive prompt
            negative_prompt: Negative prompt
            width: Image width
            height: Image height
            steps: Sampling steps
            cfg_scale: CFG guidance strength
            seed: Random seed
            **kwargs: Other parameters
            
        Returns:
            Generation result dictionary
        """
        self._ensure_initialized()
        
        self.logger.info(f"Generating image: '{prompt[:50]}...'")
        
        # Build generation parameters
        params = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'width': width,
            'height': height,
            'steps': steps,
            'cfg_scale': cfg_scale,
            'seed': seed,
            **kwargs
        }
        
        # Execute generation
        result = self.executor.execute_generation(params)
        
        self.logger.info("[OK] Generation completed")
        return result
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """
        Get available models list
        
        Returns:
            Models list dictionary
        """
        self._ensure_initialized()
        
        return {
            'checkpoints': self.model_loader.list_checkpoints(),
            'vae': self.model_loader.list_vae(),
            'loras': self.model_loader.list_loras(),
        }
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get device information
        
        Returns:
            Device information dictionary
        """
        info = {
            'device': str(self._device),
            'device_type': self._device.type if self._device else None,
        }
        
        if self._device and self._device.type == 'cuda':
            info.update({
                'gpu_name': torch.cuda.get_device_name(self.config.device_id),
                'total_vram_gb': torch.cuda.get_device_properties(self.config.device_id).total_memory / (1024**3),
                'allocated_vram_gb': torch.cuda.memory_allocated(self.config.device_id) / (1024**3),
                'reserved_vram_gb': torch.cuda.memory_reserved(self.config.device_id) / (1024**3),
            })
            
        return info
    
    def cleanup(self):
        """Cleanup resources"""
        if not self._initialized:
            return
            
        self.logger.info("Cleaning up resources...")
        
        if self.model_loader:
            self.model_loader.cleanup()
            
        if self._device and self._device.type == 'cuda':
            torch.cuda.empty_cache()
            
        self.logger.info("[OK] Cleanup completed")
    
    def _ensure_initialized(self):
        """Ensure engine is initialized"""
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
        
    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        device = str(self._device) if self._device else "unknown"
        return f"<GenesisEngine({status}, device={device})>"
