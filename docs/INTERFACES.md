Genesis Core Interfaces
======================
Author: eddy

Overview
--------
Genesis provides 4 core interfaces for maximum flexibility:
1. ComfyUI Compatibility Interface
2. Heterogeneous Computing Interface
3. Cross-Attention Interface
4. Multi-Attention Switching Interface


1. ComfyUI Compatibility Interface
-----------------------------------

Purpose: Minimal compatibility with ComfyUI node system (not full implementation)

Key Classes:
- ComfyUINodeInterface: Abstract base for ComfyUI-style nodes
- COMFYUI_NODE_REGISTRY: Global node registry
- register_comfyui_node: Decorator for node registration

Example:

```python
from genesis.core import ComfyUINodeInterface, register_comfyui_node

@register_comfyui_node("MyCustomNode")
class MyNode(ComfyUINodeInterface):
    CATEGORY = "genesis/custom"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": ""}),
                "value": ("INT", {"default": 42}),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ("OUTPUT",)
    
    def execute(self, text, value):
        result = f"{text}: {value}"
        return (result,)
```

Note: This is an interface ONLY. Not all ComfyUI features are implemented.


2. Heterogeneous Computing Interface
-------------------------------------

Purpose: Execute computations across different devices (CPU, GPU, NPU, etc.)

Key Classes:
- HeterogeneousExecutor: Manage multi-device execution
- ComputeBackend: Enum of supported backends (CUDA, CPU, NPU, MPS, ONNX, ROCM)

Example:

```python
from genesis.core import HeterogeneousExecutor, ComputeBackend
import torch

executor = HeterogeneousExecutor()

# Register backends
executor.register_backend(ComputeBackend.CUDA, torch.device('cuda:0'))
executor.register_backend(ComputeBackend.CPU, torch.device('cpu'))

# Execute on specific backend
def my_computation(x):
    return x * 2 + 1

result = executor.execute_on_backend(
    ComputeBackend.CUDA,
    my_computation,
    torch.randn(100, 100)
)

# Parallel execution across backends
results = executor.parallel_execute(
    my_computation,
    data_splits=[data1, data2],
    backends=[ComputeBackend.CPU, ComputeBackend.CUDA]
)
```

Supported Backends:
- CUDA: NVIDIA GPUs
- CPU: x86/ARM CPUs
- NPU: Huawei Ascend
- MPS: Apple Silicon
- ONNX: Cross-platform
- ROCM: AMD GPUs
- WEBGPU: Browser-based


3. Cross-Attention Interface
-----------------------------

Purpose: Cross-modal attention (e.g., image attending to text)

Key Classes:
- CrossAttentionInterface: Abstract interface
- CrossAttentionModule: Concrete implementation

Example:

```python
from genesis.core import CrossAttentionModule
import torch

# Create cross-attention module
cross_attn = CrossAttentionModule(
    query_dim=512,      # Image feature dim
    context_dim=768,    # Text embedding dim
    num_heads=8
)

# Inputs
image_features = torch.randn(2, 256, 512)  # [B, N, D]
text_embeddings = torch.randn(2, 77, 768)  # [B, M, C]

# Cross-attention: image attends to text
output = cross_attn.forward(
    query=image_features,
    key=text_embeddings,
    value=text_embeddings
)

# Get attention weights
attn_weights = cross_attn.compute_attention_weights(
    query, key, mask=None
)
```

Use Cases:
- Text-to-image generation (Stable Diffusion style)
- Image captioning
- Visual question answering
- Multimodal fusion


4. Multi-Attention Switching Interface
---------------------------------------

Purpose: Dynamically switch between different attention implementations

Key Classes:
- AdaptiveAttentionSwitch: Runtime attention switching
- AttentionType: Enum of attention types

Supported Attention Types:
- FLASH_ATTENTION: Flash Attention (fastest)
- SAGE_ATTENTION: Sage Attention
- XFORMERS: xFormers memory-efficient
- SDPA: PyTorch scaled_dot_product_attention
- PYTORCH: Standard implementation

Example:

```python
from genesis.core import AdaptiveAttentionSwitch, AttentionType
import torch

# Create adaptive attention
adaptive_attn = AdaptiveAttentionSwitch(
    embed_dim=512,
    num_heads=8,
    default_type=AttentionType.FLASH_ATTENTION
)

# Check available types
available = adaptive_attn.get_available_attentions()
print(f"Available: {[t.value for t in available]}")

# Switch attention type
adaptive_attn.set_attention_type(AttentionType.XFORMERS)

# Forward pass
x = torch.randn(2, 128, 512)
output = adaptive_attn.forward(x)

# Use specific attention for one call
output = adaptive_attn.forward_with_type(
    AttentionType.SDPA,
    query, key, value
)

# Cross-attention mode
context = torch.randn(2, 64, 512)
output = adaptive_attn.forward(x, context=context)
```

Performance Comparison (typical):
- Flash Attention: 1.0x (baseline, fastest)
- Sage Attention: 1.1x
- xFormers: 1.3x
- SDPA: 1.5x
- PyTorch: 2.5x (slowest)


Factory Function
----------------

Convenience factory for creating attention modules:

```python
from genesis.core import create_attention_interface

# Create cross-attention
cross_attn = create_attention_interface(
    'cross',
    embed_dim=512,
    context_dim=768,
    num_heads=8
)

# Create adaptive attention
adaptive_attn = create_attention_interface(
    'adaptive',
    embed_dim=512,
    num_heads=8
)
```


Interface Information
---------------------

Get runtime information about available interfaces:

```python
from genesis.core import get_interface_info

info = get_interface_info()

print(f"ComfyUI nodes: {info['comfyui']['registered_nodes']}")
print(f"Compute backends: {info['heterogeneous_compute']['backends']}")
print(f"Attention types: {info['multi_attention']['types']}")
```


Integration Example
-------------------

Combining all interfaces:

```python
from genesis.core import (
    register_comfyui_node, ComfyUINodeInterface,
    HeterogeneousExecutor, ComputeBackend,
    CrossAttentionModule,
    AdaptiveAttentionSwitch, AttentionType
)
import torch

@register_comfyui_node("GenesisMultiModalGen")
class MultiModalGenNode(ComfyUINodeInterface):
    """ComfyUI node using all advanced features"""
    
    def __init__(self):
        # Heterogeneous computing
        self.executor = HeterogeneousExecutor()
        self.executor.register_backend(ComputeBackend.CUDA, torch.device('cuda'))
        
        # Cross-attention
        self.cross_attn = CrossAttentionModule(512, 768, 8)
        
        # Adaptive attention
        self.self_attn = AdaptiveAttentionSwitch(512, 8)
        self.self_attn.set_attention_type(AttentionType.FLASH_ATTENTION)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_features": ("TENSOR",),
                "text_embeddings": ("TENSOR",),
                "backend": (["cuda", "cpu"],),
                "attention_type": (["flash", "xformers", "sdpa"],),
            }
        }
    
    def execute(self, image_features, text_embeddings, backend, attention_type):
        # Select backend
        backend_enum = ComputeBackend.CUDA if backend == "cuda" else ComputeBackend.CPU
        
        # Select attention
        attn_map = {
            "flash": AttentionType.FLASH_ATTENTION,
            "xformers": AttentionType.XFORMERS,
            "sdpa": AttentionType.SDPA
        }
        self.self_attn.set_attention_type(attn_map[attention_type])
        
        # Process with cross-attention on selected backend
        def process(img_feat, txt_emb):
            # Cross-attention: image attends to text
            cross_out = self.cross_attn.forward(img_feat, txt_emb, txt_emb)
            # Self-attention
            self_out = self.self_attn.forward(cross_out)
            return self_out
        
        result = self.executor.execute_on_backend(
            backend_enum,
            process,
            image_features,
            text_embeddings
        )
        
        return (result,)
```


Best Practices
--------------

1. ComfyUI Interface:
   - Use for compatibility layer only
   - Don't rely on full ComfyUI implementation
   - Keep nodes simple and focused

2. Heterogeneous Computing:
   - Profile different backends for your workload
   - Consider data transfer overhead
   - Use parallel execution for independent tasks

3. Cross-Attention:
   - Match dimensions carefully (query_dim, context_dim)
   - Use masking for variable-length sequences
   - Monitor attention weights for debugging

4. Multi-Attention:
   - Use Flash Attention when available (fastest)
   - Fall back to SDPA for PyTorch 2.0+
   - Benchmark on your specific hardware
   - Switch attention types based on sequence length


Performance Tips
----------------

1. Attention Selection:
   - Short sequences (<512): Any method is fine
   - Medium sequences (512-2048): Use Flash or xFormers
   - Long sequences (>2048): Flash Attention essential

2. Device Selection:
   - Model loading: CPU (avoid VRAM during load)
   - Inference: GPU (fastest)
   - Batch processing: Multi-GPU
   - Edge deployment: NPU/ONNX

3. Memory Optimization:
   - Use gradient checkpointing for training
   - Enable attention slicing for long sequences
   - Monitor VRAM usage across backends


Limitations
-----------

1. ComfyUI interface is NOT a full implementation
2. Some backends require additional dependencies
3. Attention switching has small overhead (~1-2ms)
4. Cross-platform ONNX may have accuracy differences


Dependencies
------------

Required:
- torch>=2.0.0

Optional (for full features):
- flash-attn>=2.0.0      # Flash Attention
- xformers>=0.0.22       # xFormers
- sageattention          # Sage Attention
- torch-npu              # Huawei NPU
- onnxruntime>=1.16.0    # ONNX Runtime


Running Examples
----------------

```bash
# Run interface demo
python -m genesis.examples.interfaces_demo

# Or individual demos
python -c "from genesis.examples.interfaces_demo import demo_comfyui_compatibility; demo_comfyui_compatibility()"
python -c "from genesis.examples.interfaces_demo import demo_cross_attention; demo_cross_attention()"
```


Support
-------

For issues or questions:
1. Check get_interface_info() for available features
2. Review examples in genesis/examples/interfaces_demo.py
3. Benchmark different options for your use case
