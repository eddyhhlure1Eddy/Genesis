"""
Genesis Core Interfaces
Compatibility interfaces and core abstractions
Author: eddy
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from abc import ABC, abstractmethod
from enum import Enum


# ==================== ComfyUI Compatibility Interface ====================

class ComfyUINodeInterface(ABC):
    """
    ComfyUI node compatibility interface
    Provides minimal compatibility without full implementation
    """
    
    CATEGORY = "genesis"
    
    @classmethod
    @abstractmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Define input types (ComfyUI style)
        
        Returns:
            Dictionary of input specifications
        """
        pass
    
    @classmethod
    def RETURN_TYPES(cls) -> Tuple[str, ...]:
        """Define return types"""
        return ("ANY",)
    
    @classmethod
    def RETURN_NAMES(cls) -> Tuple[str, ...]:
        """Define return names"""
        return ("output",)
    
    @classmethod
    def FUNCTION(cls) -> str:
        """Define function name to call"""
        return "execute"
    
    @abstractmethod
    def execute(self, **kwargs) -> Tuple[Any, ...]:
        """
        Execute node logic
        
        Returns:
            Tuple of outputs matching RETURN_TYPES
        """
        pass


class ComfyUICompatibleNode:
    """
    Base class for ComfyUI-compatible nodes
    Minimal implementation for interface compatibility
    """
    
    def __init__(self):
        self.outputs = {}
    
    def get_input_spec(self) -> Dict[str, Any]:
        """Get input specification"""
        return {
            "required": {},
            "optional": {},
            "hidden": {}
        }
    
    def validate_inputs(self, **kwargs) -> bool:
        """Validate input parameters"""
        return True
    
    def process(self, **kwargs) -> Any:
        """Process node logic (override this)"""
        raise NotImplementedError


class NodeRegistry:
    """
    Simple node registry for ComfyUI-style nodes
    """
    
    def __init__(self):
        self.nodes = {}
    
    def register(self, node_class: type, name: Optional[str] = None):
        """Register a node class"""
        node_name = name or node_class.__name__
        self.nodes[node_name] = node_class
        return node_class
    
    def get(self, name: str) -> Optional[type]:
        """Get node class by name"""
        return self.nodes.get(name)
    
    def list_nodes(self) -> List[str]:
        """List all registered nodes"""
        return list(self.nodes.keys())


# Global registry instance
COMFYUI_NODE_REGISTRY = NodeRegistry()


def register_comfyui_node(name: Optional[str] = None):
    """
    Decorator to register ComfyUI-compatible node
    
    Usage:
        @register_comfyui_node("MyNode")
        class MyNode(ComfyUINodeInterface):
            ...
    """
    def decorator(cls):
        COMFYUI_NODE_REGISTRY.register(cls, name)
        return cls
    return decorator


# ==================== Heterogeneous Computing Interface ====================

class ComputeBackend(Enum):
    """Compute backend types"""
    CUDA = "cuda"
    CPU = "cpu"
    NPU = "npu"
    MPS = "mps"
    ONNX = "onnx"
    ROCM = "rocm"
    WEBGPU = "webgpu"


class HeterogeneousComputeInterface(ABC):
    """
    Interface for heterogeneous computing
    Supports multiple device types and backends
    """
    
    @abstractmethod
    def get_available_backends(self) -> List[ComputeBackend]:
        """Get list of available compute backends"""
        pass
    
    @abstractmethod
    def select_backend(self, backend: ComputeBackend) -> bool:
        """
        Select compute backend
        
        Args:
            backend: Target backend
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def distribute_computation(
        self,
        computation: Callable,
        data: Any,
        devices: Optional[List[str]] = None
    ) -> Any:
        """
        Distribute computation across devices
        
        Args:
            computation: Function to execute
            data: Input data
            devices: Target devices
            
        Returns:
            Computation result
        """
        pass
    
    @abstractmethod
    def synchronize(self):
        """Synchronize all devices"""
        pass


class HeterogeneousExecutor:
    """
    Executor for heterogeneous computing
    Manages computation across different devices
    """
    
    def __init__(self):
        self.backends = {}
        self.current_backend = None
        self.device_map = {}
    
    def register_backend(self, backend: ComputeBackend, device: torch.device):
        """Register a compute backend"""
        self.backends[backend] = device
    
    def execute_on_backend(
        self,
        backend: ComputeBackend,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function on specific backend
        
        Args:
            backend: Target backend
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Execution result
        """
        if backend not in self.backends:
            raise ValueError(f"Backend {backend} not available")
        
        device = self.backends[backend]
        
        # Move data to device
        args_on_device = [
            arg.to(device) if isinstance(arg, torch.Tensor) else arg
            for arg in args
        ]
        
        # Execute
        result = func(*args_on_device, **kwargs)
        
        return result
    
    def parallel_execute(
        self,
        func: Callable,
        data_splits: List[Any],
        backends: Optional[List[ComputeBackend]] = None
    ) -> List[Any]:
        """
        Execute function in parallel across backends
        
        Args:
            func: Function to execute
            data_splits: List of data splits
            backends: Target backends
            
        Returns:
            List of results
        """
        if backends is None:
            backends = list(self.backends.keys())
        
        results = []
        for data, backend in zip(data_splits, backends):
            result = self.execute_on_backend(backend, func, data)
            results.append(result)
        
        return results


# ==================== Cross Attention Interface ====================

class CrossAttentionInterface(ABC):
    """
    Interface for cross-attention mechanisms
    Used for conditional generation (e.g., text-to-image)
    """
    
    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Cross-attention forward pass
        
        Args:
            query: Query tensor from target modality
            key: Key tensor from source modality (e.g., text)
            value: Value tensor from source modality
            context: Optional context tensor
            mask: Optional attention mask
            
        Returns:
            Cross-attention output
        """
        pass
    
    @abstractmethod
    def compute_attention_weights(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention weights
        
        Args:
            query: Query tensor
            key: Key tensor
            mask: Optional mask
            
        Returns:
            Attention weights [B, H, N, M]
        """
        pass


class CrossAttentionModule(nn.Module, CrossAttentionInterface):
    """
    Flexible cross-attention implementation
    """
    
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        
        # Projections
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, self.inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
        self.scale = head_dim ** -0.5
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Cross-attention forward"""
        batch_size = query.shape[0]
        
        # Use context if provided, otherwise use key/value
        if context is not None:
            key = context
            value = context
        
        # Project
        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)
        
        # Reshape for multi-head
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention weights
        attn_weights = self.compute_attention_weights(q, k, mask)
        
        # Apply attention
        out = torch.matmul(attn_weights, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.inner_dim)
        out = self.to_out(out)
        
        return out
    
    def compute_attention_weights(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute attention weights"""
        attn = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(attn, dim=-1)
        return attn


# ==================== Multi-Attention Switching Interface ====================

class AttentionType(Enum):
    """Attention mechanism types"""
    SELF_ATTENTION = "self"
    CROSS_ATTENTION = "cross"
    FLASH_ATTENTION = "flash"
    SAGE_ATTENTION = "sage"
    XFORMERS = "xformers"
    SDPA = "sdpa"
    PYTORCH = "pytorch"


class MultiAttentionInterface(ABC):
    """
    Interface for switching between different attention mechanisms
    """
    
    @abstractmethod
    def set_attention_type(self, attn_type: AttentionType) -> bool:
        """
        Switch attention mechanism
        
        Args:
            attn_type: Target attention type
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def get_available_attentions(self) -> List[AttentionType]:
        """Get list of available attention types"""
        pass
    
    @abstractmethod
    def forward_with_type(
        self,
        attn_type: AttentionType,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with specific attention type
        
        Args:
            attn_type: Attention type to use
            query: Query tensor
            key: Key tensor
            value: Value tensor
            **kwargs: Additional arguments
            
        Returns:
            Attention output
        """
        pass


class AdaptiveAttentionSwitch(nn.Module, MultiAttentionInterface):
    """
    Adaptive attention switcher
    Dynamically switches between attention mechanisms
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        default_type: AttentionType = AttentionType.SDPA
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.current_type = default_type
        
        # Import attention implementations
        self._load_attention_backends()
        
        # Projections
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def _load_attention_backends(self):
        """Load available attention backends"""
        self.backends = {}
        
        # Check Flash Attention
        try:
            from flash_attn import flash_attn_func
            self.backends[AttentionType.FLASH_ATTENTION] = flash_attn_func
        except ImportError:
            pass
        
        # Check Sage Attention
        try:
            import sageattention
            self.backends[AttentionType.SAGE_ATTENTION] = sageattention
        except ImportError:
            pass
        
        # Check xFormers
        try:
            import xformers.ops as xops
            self.backends[AttentionType.XFORMERS] = xops.memory_efficient_attention
        except ImportError:
            pass
        
        # SDPA (PyTorch 2.0+)
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            self.backends[AttentionType.SDPA] = torch.nn.functional.scaled_dot_product_attention
        
        # PyTorch (always available)
        self.backends[AttentionType.PYTORCH] = self._pytorch_attention
    
    def _pytorch_attention(self, q, k, v, **kwargs):
        """Standard PyTorch attention"""
        scale = self.embed_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, v)
    
    def set_attention_type(self, attn_type: AttentionType) -> bool:
        """Switch attention type"""
        if attn_type in self.backends:
            self.current_type = attn_type
            return True
        return False
    
    def get_available_attentions(self) -> List[AttentionType]:
        """Get available attention types"""
        return list(self.backends.keys())
    
    def forward_with_type(
        self,
        attn_type: AttentionType,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Forward with specific attention type"""
        if attn_type not in self.backends:
            raise ValueError(f"Attention type {attn_type} not available")
        
        backend = self.backends[attn_type]
        return backend(query, key, value, **kwargs)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with current attention type
        
        Args:
            x: Input tensor [B, N, D]
            context: Context for cross-attention
            mask: Attention mask
            
        Returns:
            Output tensor [B, N, D]
        """
        batch_size = x.shape[0]
        
        # Self-attention
        if context is None:
            qkv = self.qkv_proj(x)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            # Cross-attention
            q = self.qkv_proj(x)[:, :, :self.embed_dim]
            kv = self.qkv_proj(context)[:, :, self.embed_dim:]
            k, v = kv.chunk(2, dim=-1)
        
        # Reshape for multi-head
        head_dim = self.embed_dim // self.num_heads
        q = q.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        
        # Apply attention
        out = self.forward_with_type(self.current_type, q, k, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.embed_dim)
        out = self.out_proj(out)
        
        return out


# ==================== Utility Functions ====================

def create_attention_interface(
    attention_type: str,
    embed_dim: int,
    num_heads: int = 8,
    **kwargs
) -> nn.Module:
    """
    Factory function to create attention interface
    
    Args:
        attention_type: 'self', 'cross', or 'adaptive'
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        **kwargs: Additional arguments
        
    Returns:
        Attention module
    """
    if attention_type == 'cross':
        context_dim = kwargs.get('context_dim', embed_dim)
        return CrossAttentionModule(embed_dim, context_dim, num_heads)
    
    elif attention_type == 'adaptive':
        return AdaptiveAttentionSwitch(embed_dim, num_heads)
    
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


def get_interface_info() -> Dict[str, Any]:
    """Get information about available interfaces"""
    return {
        'comfyui': {
            'available': True,
            'registered_nodes': len(COMFYUI_NODE_REGISTRY.list_nodes())
        },
        'heterogeneous_compute': {
            'available': True,
            'backends': [b.value for b in ComputeBackend]
        },
        'cross_attention': {
            'available': True
        },
        'multi_attention': {
            'available': True,
            'types': [t.value for t in AttentionType]
        }
    }
