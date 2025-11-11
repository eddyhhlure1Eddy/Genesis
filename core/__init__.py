"""Genesis Core - Core Engine Module"""

from .engine import GenesisEngine
from .pipeline import Pipeline, PipelineBuilder
from .config import GenesisConfig
from .nodes import NODE_REGISTRY, BaseNode
from .samplers import Sampler, SamplerRegistry
from .vae import VAE, VAEProcessor
from .clip import CLIPTextEncoder, PromptParser
from .acceleration import (
    DeviceManager, AccelerationEngine, MemoryManager,
    TensorOptimizer, get_optimal_dtype, benchmark_operation
)
from .triton_kernels import TritonOps, get_triton_config
from .multi_backend import (
    MultiDeviceManager, NPUAccelerator, ONNXBackend,
    MultiGPUManager, BackendType, get_backend_info
)
from .attention import (
    UnifiedAttention, MultiHeadAttention, AttentionBackend,
    benchmark_attention_backends, get_attention_info
)
from .async_engine import (
    AsyncModelLoader, AsyncInferenceEngine, AsyncBatchProcessor,
    async_model_warmup
)
from .interfaces import (
    ComfyUINodeInterface, ComfyUICompatibleNode, NodeRegistry,
    COMFYUI_NODE_REGISTRY, register_comfyui_node,
    HeterogeneousComputeInterface, HeterogeneousExecutor, ComputeBackend,
    CrossAttentionInterface, CrossAttentionModule,
    MultiAttentionInterface, AdaptiveAttentionSwitch, AttentionType,
    create_attention_interface, get_interface_info
)

__all__ = [
    # Engine
    'GenesisEngine',
    'Pipeline',
    'PipelineBuilder',
    'GenesisConfig',
    'NODE_REGISTRY',
    'BaseNode',
    
    # Samplers
    'Sampler',
    'SamplerRegistry',
    
    # VAE & CLIP
    'VAE',
    'VAEProcessor',
    'CLIPTextEncoder',
    'PromptParser',
    
    # Acceleration
    'DeviceManager',
    'AccelerationEngine',
    'MemoryManager',
    'TensorOptimizer',
    'get_optimal_dtype',
    'benchmark_operation',
    
    # Triton
    'TritonOps',
    'get_triton_config',
    
    # Multi-Backend
    'MultiDeviceManager',
    'NPUAccelerator',
    'ONNXBackend',
    'MultiGPUManager',
    'BackendType',
    'get_backend_info',
    
    # Attention
    'UnifiedAttention',
    'MultiHeadAttention',
    'AttentionBackend',
    'benchmark_attention_backends',
    'get_attention_info',
    
    # Async
    'AsyncModelLoader',
    'AsyncInferenceEngine',
    'AsyncBatchProcessor',
    'async_model_warmup',
    
    # Interfaces
    'ComfyUINodeInterface',
    'ComfyUICompatibleNode',
    'NodeRegistry',
    'COMFYUI_NODE_REGISTRY',
    'register_comfyui_node',
    'HeterogeneousComputeInterface',
    'HeterogeneousExecutor',
    'ComputeBackend',
    'CrossAttentionInterface',
    'CrossAttentionModule',
    'MultiAttentionInterface',
    'AdaptiveAttentionSwitch',
    'AttentionType',
    'create_attention_interface',
    'get_interface_info'
]
