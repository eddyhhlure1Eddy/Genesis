"""
Genesis Acceleration Core
PyTorch acceleration, optimization, and device management
Author: eddy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Tuple, Dict, Any
import logging
import platform
import subprocess


class DeviceManager:
    """
    Advanced device management and acceleration
    """
    
    def __init__(self):
        self.logger = logging.getLogger('Genesis.DeviceManager')
        self.device = None
        self.device_type = None
        self.device_name = None
        self.compute_capability = None
        self.total_memory = 0
        self.available_memory = 0
        
        self._detect_device()
        self._setup_optimizations()
    
    def _detect_device(self):
        """Detect best available device"""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.device_type = 'cuda'
            self.device_name = torch.cuda.get_device_name(0)
            
            props = torch.cuda.get_device_properties(0)
            self.compute_capability = (props.major, props.minor)
            self.total_memory = props.total_memory
            self.available_memory = torch.cuda.mem_get_info()[0]
            
            self.logger.info(f"Using CUDA: {self.device_name}")
            self.logger.info(f"Compute Capability: {self.compute_capability}")
            self.logger.info(f"Total Memory: {self.total_memory / 1e9:.2f} GB")
            
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            self.device_type = 'mps'
            self.device_name = 'Apple Silicon'
            self.logger.info(f"Using MPS: {self.device_name}")
            
        else:
            self.device = torch.device('cpu')
            self.device_type = 'cpu'
            self.device_name = platform.processor() or 'CPU'
            self.logger.info(f"Using CPU: {self.device_name}")
    
    def _setup_optimizations(self):
        """Setup device-specific optimizations"""
        if self.device_type == 'cuda':
            # Enable TF32 for Ampere+ GPUs
            if self.compute_capability and self.compute_capability[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                self.logger.info("TF32 enabled for faster computation")
            
            # Benchmark cuDNN algorithms
            torch.backends.cudnn.benchmark = True
            self.logger.info("cuDNN benchmark enabled")
            
            # Enable deterministic algorithms if needed
            # torch.use_deterministic_algorithms(True)
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get device memory information"""
        if self.device_type == 'cuda':
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            free, total = torch.cuda.mem_get_info(0)
            
            return {
                'allocated_gb': allocated / 1e9,
                'reserved_gb': reserved / 1e9,
                'free_gb': free / 1e9,
                'total_gb': total / 1e9,
                'utilization': (allocated / total) * 100
            }
        
        return {}
    
    def empty_cache(self):
        """Clear GPU cache"""
        if self.device_type == 'cuda':
            torch.cuda.empty_cache()
            self.logger.debug("GPU cache cleared")
    
    def synchronize(self):
        """Synchronize device"""
        if self.device_type == 'cuda':
            torch.cuda.synchronize()


class AccelerationEngine:
    """
    Main acceleration engine
    Integrates various optimization techniques
    """
    
    def __init__(self, device_manager: DeviceManager):
        self.device_manager = device_manager
        self.device = device_manager.device
        self.logger = logging.getLogger('Genesis.Acceleration')
        
        # Check for acceleration libraries
        self.xformers_available = self._check_xformers()
        self.triton_available = self._check_triton()
        self.flash_attn_available = self._check_flash_attention()
        
        self.logger.info(f"Acceleration libraries:")
        self.logger.info(f"  - xFormers: {self.xformers_available}")
        self.logger.info(f"  - Triton: {self.triton_available}")
        self.logger.info(f"  - Flash Attention: {self.flash_attn_available}")
    
    def _check_xformers(self) -> bool:
        """Check if xFormers is available"""
        try:
            import xformers
            import xformers.ops
            return True
        except ImportError:
            return False
    
    def _check_triton(self) -> bool:
        """Check if Triton is available"""
        try:
            import triton
            return True
        except ImportError:
            return False
    
    def _check_flash_attention(self) -> bool:
        """Check if Flash Attention is available"""
        try:
            from flash_attn import flash_attn_func
            return True
        except ImportError:
            return False
    
    def optimize_model(self, model: nn.Module, mode: str = 'default') -> nn.Module:
        """
        Optimize model for inference
        
        Args:
            model: PyTorch model
            mode: Optimization mode ('default', 'aggressive', 'memory')
        
        Returns:
            Optimized model
        """
        model = model.to(self.device)
        model.eval()
        
        if mode == 'aggressive':
            # Use torch.compile if available (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, mode='max-autotune')
                    self.logger.info("Model compiled with torch.compile")
                except Exception as e:
                    self.logger.warning(f"Failed to compile model: {e}")
        
        elif mode == 'memory':
            # Enable gradient checkpointing for memory efficiency
            if hasattr(model, 'enable_gradient_checkpointing'):
                model.enable_gradient_checkpointing()
                self.logger.info("Gradient checkpointing enabled")
        
        return model
    
    def autocast_context(self, dtype: Optional[torch.dtype] = None):
        """
        Get autocast context for mixed precision
        
        Args:
            dtype: Target dtype (default: bfloat16 for Ampere+, float16 otherwise)
        
        Returns:
            Autocast context manager
        """
        if self.device.type == 'cuda':
            if dtype is None:
                # Use bfloat16 for Ampere+ GPUs
                if self.device_manager.compute_capability and self.device_manager.compute_capability[0] >= 8:
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float16
            
            return torch.cuda.amp.autocast(dtype=dtype)
        
        elif self.device.type == 'cpu':
            return torch.cpu.amp.autocast(dtype=torch.bfloat16)
        
        else:
            # Fallback: no-op context
            from contextlib import nullcontext
            return nullcontext()
    
    def efficient_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Efficient attention computation
        Uses Flash Attention, xFormers, or fallback
        
        Args:
            query: Query tensor [B, H, N, D]
            key: Key tensor [B, H, N, D]
            value: Value tensor [B, H, N, D]
            attn_mask: Attention mask
            dropout_p: Dropout probability
            scale: Attention scale
        
        Returns:
            Attention output [B, H, N, D]
        """
        # Try Flash Attention first (fastest)
        if self.flash_attn_available and attn_mask is None:
            try:
                from flash_attn import flash_attn_func
                
                # Flash attention expects [B, N, H, D]
                q = query.transpose(1, 2)
                k = key.transpose(1, 2)
                v = value.transpose(1, 2)
                
                out = flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=scale)
                return out.transpose(1, 2)
            
            except Exception as e:
                self.logger.debug(f"Flash attention failed: {e}")
        
        # Try xFormers memory efficient attention
        if self.xformers_available:
            try:
                import xformers.ops as xops
                
                out = xops.memory_efficient_attention(
                    query, key, value,
                    attn_bias=attn_mask,
                    p=dropout_p,
                    scale=scale
                )
                return out
            
            except Exception as e:
                self.logger.debug(f"xFormers attention failed: {e}")
        
        # Fallback to PyTorch scaled_dot_product_attention
        if hasattr(F, 'scaled_dot_product_attention'):
            return F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                scale=scale
            )
        
        # Manual attention (slowest, most memory intensive)
        if scale is None:
            scale = 1.0 / (query.size(-1) ** 0.5)
        
        attn = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        if attn_mask is not None:
            attn = attn + attn_mask
        
        attn = F.softmax(attn, dim=-1)
        
        if dropout_p > 0:
            attn = F.dropout(attn, p=dropout_p)
        
        out = torch.matmul(attn, value)
        return out


class MemoryManager:
    """
    Advanced memory management
    """
    
    def __init__(self, device_manager: DeviceManager):
        self.device_manager = device_manager
        self.device = device_manager.device
        self.logger = logging.getLogger('Genesis.Memory')
    
    def estimate_model_memory(self, model: nn.Module) -> Dict[str, float]:
        """
        Estimate model memory usage
        
        Returns:
            Dictionary with memory estimates in GB
        """
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
        
        return {
            'parameters_gb': param_memory / 1e9,
            'buffers_gb': buffer_memory / 1e9,
            'total_gb': (param_memory + buffer_memory) / 1e9
        }
    
    def optimize_memory_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Optimize tensor memory layout
        """
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        return tensor
    
    def pin_memory(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Pin tensor to memory for faster CPU->GPU transfer
        """
        if self.device.type == 'cuda' and tensor.device.type == 'cpu':
            return tensor.pin_memory()
        return tensor


class TensorOptimizer:
    """
    Tensor operation optimizations
    """
    
    def __init__(self, device: torch.device):
        self.device = device
    
    @staticmethod
    def fused_matmul_add(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Fused matrix multiplication and addition"""
        return torch.addmm(c, a, b)
    
    @staticmethod
    def fused_linear(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fused linear layer"""
        return F.linear(input, weight, bias)
    
    @staticmethod
    def efficient_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Memory efficient softmax"""
        x_max = torch.max(x, dim=dim, keepdim=True)[0]
        exp_x = torch.exp(x - x_max)
        return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
    
    @staticmethod
    def efficient_layernorm(x: torch.Tensor, normalized_shape: Union[int, List[int]], 
                           weight: Optional[torch.Tensor] = None, 
                           bias: Optional[torch.Tensor] = None,
                           eps: float = 1e-5) -> torch.Tensor:
        """Efficient layer normalization"""
        return F.layer_norm(x, normalized_shape, weight, bias, eps)


def get_optimal_dtype(device: torch.device, compute_capability: Optional[Tuple[int, int]] = None) -> torch.dtype:
    """
    Get optimal dtype for device
    
    Args:
        device: Target device
        compute_capability: GPU compute capability (major, minor)
    
    Returns:
        Optimal dtype
    """
    if device.type == 'cuda':
        # Use bfloat16 for Ampere+ (8.0+)
        if compute_capability and compute_capability[0] >= 8:
            return torch.bfloat16
        # Use float16 for older GPUs
        return torch.float16
    
    elif device.type == 'cpu':
        # Use bfloat16 for CPU
        return torch.bfloat16
    
    else:
        # Default to float32
        return torch.float32


def benchmark_operation(func, *args, num_iterations: int = 100, warmup: int = 10) -> float:
    """
    Benchmark operation performance
    
    Args:
        func: Function to benchmark
        *args: Function arguments
        num_iterations: Number of iterations
        warmup: Warmup iterations
    
    Returns:
        Average time in milliseconds
    """
    import time
    
    # Warmup
    for _ in range(warmup):
        func(*args)
    
    # Synchronize if CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        func(*args)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end = time.perf_counter()
    
    avg_time = ((end - start) / num_iterations) * 1000  # Convert to ms
    return avg_time
