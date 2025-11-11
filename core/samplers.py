"""
Genesis Samplers
Sampling algorithms for diffusion models
Author: eddy
"""

import torch
import numpy as np
from typing import Optional, Callable, Dict, Any
from enum import Enum


class SamplerType(Enum):
    """Available sampler types"""
    EULER = "euler"
    EULER_A = "euler_a"
    HEUN = "heun"
    DPM_2 = "dpm_2"
    DPM_2_A = "dpm_2_a"
    DPM_PP_2S_A = "dpm_pp_2s_a"
    DPM_PP_2M = "dpm_pp_2m"
    DPM_PP_SDE = "dpm_pp_sde"
    LMS = "lms"
    DDIM = "ddim"
    DDPM = "ddpm"


class SchedulerType(Enum):
    """Available scheduler types"""
    NORMAL = "normal"
    KARRAS = "karras"
    EXPONENTIAL = "exponential"
    SGM_UNIFORM = "sgm_uniform"
    SIMPLE = "simple"
    DDIM_UNIFORM = "ddim_uniform"


class Sampler:
    """
    Base sampler class
    
    Handles the sampling process for diffusion models
    """
    
    def __init__(
        self,
        sampler_type: str = "euler",
        scheduler_type: str = "normal",
        steps: int = 20,
        cfg_scale: float = 7.0,
        denoise: float = 1.0
    ):
        """
        Initialize sampler
        
        Args:
            sampler_type: Sampler algorithm
            scheduler_type: Noise scheduler type
            steps: Number of sampling steps
            cfg_scale: Classifier-free guidance scale
            denoise: Denoising strength (1.0 = full denoise)
        """
        self.sampler_type = sampler_type
        self.scheduler_type = scheduler_type
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.denoise = denoise
        
    def get_sigmas(self, steps: int, scheduler: str = "normal") -> torch.Tensor:
        """
        Get noise schedule sigmas
        
        Args:
            steps: Number of steps
            scheduler: Scheduler type
            
        Returns:
            Sigma values tensor
        """
        if scheduler == "karras":
            return self._get_sigmas_karras(steps)
        elif scheduler == "exponential":
            return self._get_sigmas_exponential(steps)
        else:
            return self._get_sigmas_normal(steps)
    
    def _get_sigmas_normal(self, steps: int) -> torch.Tensor:
        """Normal/linear sigma schedule"""
        sigma_min = 0.0292
        sigma_max = 14.6146
        
        sigmas = torch.linspace(
            np.log(sigma_max),
            np.log(sigma_min),
            steps
        ).exp()
        
        return torch.cat([sigmas, torch.zeros(1)])
    
    def _get_sigmas_karras(self, steps: int) -> torch.Tensor:
        """Karras sigma schedule (better quality)"""
        sigma_min = 0.0292
        sigma_max = 14.6146
        rho = 7.0
        
        ramp = torch.linspace(0, 1, steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        
        return torch.cat([sigmas, torch.zeros(1)])
    
    def _get_sigmas_exponential(self, steps: int) -> torch.Tensor:
        """Exponential sigma schedule"""
        sigma_min = 0.0292
        sigma_max = 14.6146
        
        sigmas = torch.linspace(
            np.log(sigma_max),
            np.log(sigma_min),
            steps
        ).exp()
        
        return torch.cat([sigmas, torch.zeros(1)])
    
    def sample(
        self,
        model: Any,
        latent: torch.Tensor,
        positive_conditioning: torch.Tensor,
        negative_conditioning: torch.Tensor,
        seed: Optional[int] = None,
        callback: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        Perform sampling
        
        Args:
            model: Diffusion model
            latent: Initial latent tensor
            positive_conditioning: Positive prompt conditioning
            negative_conditioning: Negative prompt conditioning
            seed: Random seed
            callback: Progress callback function
            
        Returns:
            Sampled latent tensor
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Get sigmas
        sigmas = self.get_sigmas(self.steps, self.scheduler_type)
        
        # Add noise to latent
        noise = torch.randn_like(latent)
        x = latent + noise * sigmas[0]
        
        # Sampling loop
        for i in range(self.steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            # Model prediction with CFG
            if self.cfg_scale != 1.0:
                # Unconditional prediction
                uncond_pred = model(x, sigma, negative_conditioning)
                # Conditional prediction
                cond_pred = model(x, sigma, positive_conditioning)
                # CFG
                pred = uncond_pred + self.cfg_scale * (cond_pred - uncond_pred)
            else:
                pred = model(x, sigma, positive_conditioning)
            
            # Update x based on sampler type
            if self.sampler_type == "euler":
                x = self._euler_step(x, pred, sigma, sigma_next)
            elif self.sampler_type == "euler_a":
                x = self._euler_ancestral_step(x, pred, sigma, sigma_next)
            elif self.sampler_type == "heun":
                x = self._heun_step(x, pred, sigma, sigma_next, model, positive_conditioning)
            else:
                x = self._euler_step(x, pred, sigma, sigma_next)
            
            # Progress callback
            if callback:
                callback(i + 1, self.steps)
        
        return x
    
    def _euler_step(
        self,
        x: torch.Tensor,
        pred: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor
    ) -> torch.Tensor:
        """Euler method step"""
        dt = sigma_next - sigma
        return x + pred * dt
    
    def _euler_ancestral_step(
        self,
        x: torch.Tensor,
        pred: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor
    ) -> torch.Tensor:
        """Euler ancestral method step (adds noise)"""
        dt = sigma_next - sigma
        x = x + pred * dt
        
        if sigma_next > 0:
            noise = torch.randn_like(x)
            x = x + noise * sigma_next
        
        return x
    
    def _heun_step(
        self,
        x: torch.Tensor,
        pred: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        model: Any,
        conditioning: torch.Tensor
    ) -> torch.Tensor:
        """Heun's method step (2nd order)"""
        dt = sigma_next - sigma
        
        # First step
        x_2 = x + pred * dt
        
        # Second prediction
        pred_2 = model(x_2, sigma_next, conditioning)
        
        # Average
        pred_avg = (pred + pred_2) / 2
        
        return x + pred_avg * dt


class SamplerRegistry:
    """Registry for available samplers"""
    
    SAMPLERS = {
        "euler": "Euler",
        "euler_a": "Euler Ancestral",
        "heun": "Heun",
        "dpm_2": "DPM 2",
        "dpm_2_a": "DPM 2 Ancestral",
        "dpm_pp_2s_a": "DPM++ 2S Ancestral",
        "dpm_pp_2m": "DPM++ 2M",
        "dpm_pp_sde": "DPM++ SDE",
        "lms": "LMS",
        "ddim": "DDIM",
        "ddpm": "DDPM",
    }
    
    SCHEDULERS = {
        "normal": "Normal",
        "karras": "Karras",
        "exponential": "Exponential",
        "sgm_uniform": "SGM Uniform",
        "simple": "Simple",
        "ddim_uniform": "DDIM Uniform",
    }
    
    @classmethod
    def get_sampler_names(cls) -> list:
        """Get list of available sampler names"""
        return list(cls.SAMPLERS.keys())
    
    @classmethod
    def get_scheduler_names(cls) -> list:
        """Get list of available scheduler names"""
        return list(cls.SCHEDULERS.keys())
    
    @classmethod
    def create_sampler(
        cls,
        sampler_name: str,
        scheduler_name: str = "normal",
        **kwargs
    ) -> Sampler:
        """
        Create sampler instance
        
        Args:
            sampler_name: Sampler type name
            scheduler_name: Scheduler type name
            **kwargs: Additional sampler parameters
            
        Returns:
            Sampler instance
        """
        if sampler_name not in cls.SAMPLERS:
            raise ValueError(f"Unknown sampler: {sampler_name}")
        
        if scheduler_name not in cls.SCHEDULERS:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
        return Sampler(
            sampler_type=sampler_name,
            scheduler_type=scheduler_name,
            **kwargs
        )
