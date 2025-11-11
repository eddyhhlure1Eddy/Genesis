#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Genesis Main Entry Point
Real production entry point with PyTorch acceleration
Author: eddy
"""

import sys
import argparse
import logging
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from genesis import GenesisEngine
from genesis.core.config import GenesisConfig
from genesis.core.acceleration import (
    DeviceManager, AccelerationEngine, MemoryManager, 
    TensorOptimizer, get_optimal_dtype, benchmark_operation
)
from genesis.core.triton_kernels import get_triton_config, TritonOps
import torch
import torch.nn as nn


def setup_logging(level: str = 'INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def show_system_info():
    """Display system and acceleration information"""
    print("\n" + "="*70)
    print(" Genesis System Information")
    print("="*70)
    
    # Device Manager
    device_mgr = DeviceManager()
    print(f"\nDevice:")
    print(f"  Type: {device_mgr.device_type}")
    print(f"  Name: {device_mgr.device_name}")
    
    if device_mgr.device_type == 'cuda':
        print(f"  Compute Capability: {device_mgr.compute_capability}")
        print(f"  Total Memory: {device_mgr.total_memory / 1e9:.2f} GB")
        
        mem_info = device_mgr.get_memory_info()
        if mem_info:
            print(f"\nMemory:")
            print(f"  Allocated: {mem_info['allocated_gb']:.2f} GB")
            print(f"  Free: {mem_info['free_gb']:.2f} GB")
            print(f"  Utilization: {mem_info['utilization']:.1f}%")
    
    # Acceleration Engine
    accel_engine = AccelerationEngine(device_mgr)
    print(f"\nAcceleration:")
    print(f"  xFormers: {'Yes' if accel_engine.xformers_available else 'No'}")
    print(f"  Triton: {'Yes' if accel_engine.triton_available else 'No'}")
    print(f"  Flash Attention: {'Yes' if accel_engine.flash_attn_available else 'No'}")
    
    # Triton config
    triton_config = get_triton_config()
    if triton_config['available']:
        print(f"\nTriton:")
        print(f"  Version: {triton_config.get('version', 'unknown')}")
        if 'device' in triton_config:
            print(f"  Device: {triton_config['device']}")
            print(f"  Compute Capability: {triton_config['compute_capability']}")
    
    # PyTorch
    print(f"\nPyTorch:")
    print(f"  Version: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"  cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
    
    print("\n" + "="*70)


def benchmark_acceleration():
    """Benchmark acceleration features"""
    print("\n" + "="*70)
    print(" Acceleration Benchmarks")
    print("="*70)
    
    device_mgr = DeviceManager()
    device = device_mgr.device
    
    # Test tensor size
    size = (1024, 1024)
    
    # Create test tensors
    x = torch.randn(size, device=device)
    y = torch.randn(size, device=device)
    
    print(f"\nTest tensor size: {size}")
    print(f"Device: {device}")
    
    # Benchmark addition
    print(f"\n1. Element-wise Addition:")
    
    # PyTorch
    torch_time = benchmark_operation(lambda: x + y, num_iterations=100)
    print(f"   PyTorch: {torch_time:.3f} ms")
    
    # Triton (if available)
    if get_triton_config()['available'] and device.type == 'cuda':
        triton_time = benchmark_operation(lambda: TritonOps.add(x, y), num_iterations=100)
        print(f"   Triton:  {triton_time:.3f} ms")
        speedup = torch_time / triton_time
        print(f"   Speedup: {speedup:.2f}x")
    
    # Benchmark softmax
    print(f"\n2. Softmax:")
    x_2d = torch.randn(512, 2048, device=device)
    
    torch_time = benchmark_operation(lambda: torch.softmax(x_2d, dim=-1), num_iterations=50)
    print(f"   PyTorch: {torch_time:.3f} ms")
    
    if get_triton_config()['available'] and device.type == 'cuda':
        triton_time = benchmark_operation(lambda: TritonOps.softmax(x_2d), num_iterations=50)
        print(f"   Triton:  {triton_time:.3f} ms")
        speedup = torch_time / triton_time
        print(f"   Speedup: {speedup:.2f}x")
    
    # Benchmark attention
    print(f"\n3. Attention (if available):")
    accel_engine = AccelerationEngine(device_mgr)
    
    # Smaller size for attention
    batch, heads, seq_len, dim = 2, 8, 512, 64
    q = torch.randn(batch, heads, seq_len, dim, device=device)
    k = torch.randn(batch, heads, seq_len, dim, device=device)
    v = torch.randn(batch, heads, seq_len, dim, device=device)
    
    if accel_engine.xformers_available or accel_engine.flash_attn_available:
        attn_time = benchmark_operation(
            lambda: accel_engine.efficient_attention(q, k, v),
            num_iterations=20
        )
        print(f"   Efficient Attention: {attn_time:.3f} ms")
    else:
        print(f"   No accelerated attention available")
    
    print("\n" + "="*70)


def demonstrate_mixed_precision():
    """Demonstrate mixed precision training"""
    print("\n" + "="*70)
    print(" Mixed Precision Demo")
    print("="*70)
    
    device_mgr = DeviceManager()
    accel_engine = AccelerationEngine(device_mgr)
    device = device_mgr.device
    
    # Simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(1024, 2048)
            self.linear2 = nn.Linear(2048, 1024)
        
        def forward(self, x):
            x = torch.relu(self.linear1(x))
            x = self.linear2(x)
            return x
    
    model = SimpleModel().to(device)
    x = torch.randn(32, 1024, device=device)
    
    # FP32
    print(f"\n1. FP32 (default):")
    start = time.perf_counter()
    for _ in range(10):
        _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    fp32_time = (time.perf_counter() - start) * 1000
    print(f"   Time: {fp32_time:.2f} ms")
    
    # Mixed Precision
    print(f"\n2. Mixed Precision (autocast):")
    with accel_engine.autocast_context():
        start = time.perf_counter()
        for _ in range(10):
            _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        mixed_time = (time.perf_counter() - start) * 1000
    print(f"   Time: {mixed_time:.2f} ms")
    print(f"   Speedup: {fp32_time / mixed_time:.2f}x")
    
    print("\n" + "="*70)


def show_memory_optimization():
    """Demonstrate memory optimization"""
    print("\n" + "="*70)
    print(" Memory Optimization Demo")
    print("="*70)
    
    device_mgr = DeviceManager()
    memory_mgr = MemoryManager(device_mgr)
    
    if device_mgr.device_type != 'cuda':
        print("\nMemory optimization demos require CUDA")
        return
    
    # Show initial memory
    mem_info = device_mgr.get_memory_info()
    print(f"\nInitial Memory:")
    print(f"  Allocated: {mem_info['allocated_gb']:.2f} GB")
    print(f"  Free: {mem_info['free_gb']:.2f} GB")
    
    # Allocate large tensor
    print(f"\nAllocating 1GB tensor...")
    large_tensor = torch.randn(256, 1024, 1024, device=device_mgr.device)
    
    mem_info = device_mgr.get_memory_info()
    print(f"After allocation:")
    print(f"  Allocated: {mem_info['allocated_gb']:.2f} GB")
    print(f"  Free: {mem_info['free_gb']:.2f} GB")
    
    # Free tensor
    print(f"\nFreeing tensor...")
    del large_tensor
    device_mgr.empty_cache()
    
    mem_info = device_mgr.get_memory_info()
    print(f"After cleanup:")
    print(f"  Allocated: {mem_info['allocated_gb']:.2f} GB")
    print(f"  Free: {mem_info['free_gb']:.2f} GB")
    
    print("\n" + "="*70)


def demo_basic():
    """Basic demo"""
    print("=" * 60)
    print("Genesis Engine - Basic Demo")
    print("=" * 60)
    
    # Create config
    config = GenesisConfig(
        device='cuda',
        log_level='INFO'
    )
    
    # Create engine
    engine = GenesisEngine(config)
    
    # Initialize
    engine.initialize()
    
    # Get device info
    device_info = engine.get_device_info()
    print(f"\nDevice Info:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    # Get available models
    print(f"\nAvailable Models:")
    try:
        models = engine.get_available_models()
        for model_type, model_list in models.items():
            print(f"  {model_type}: {len(model_list)} models")
            if model_list:
                print(f"    - {model_list[0]}")
    except Exception as e:
        print(f"  Cannot list models: {e}")
    
    # Cleanup
    engine.cleanup()
    
    print("\nDemo completed")


def demo_pipeline():
    """Pipeline demo"""
    print("\n" + "=" * 60)
    print("Genesis Pipeline - Workflow Demo")
    print("=" * 60)
    
    # Create Pipeline
    pipeline = Pipeline("demo_workflow")
    
    # Add nodes
    loader = pipeline.add_node("CheckpointLoader", checkpoint="model.safetensors")
    sampler = pipeline.add_node("KSampler", steps=20, cfg=7.0)
    decoder = pipeline.add_node("VAEDecode")
    save = pipeline.add_node("SaveImage", filename_prefix="genesis")
    
    # Connect nodes
    pipeline.connect(loader, "MODEL", sampler, "model")
    pipeline.connect(sampler, "LATENT", decoder, "samples")
    pipeline.connect(decoder, "IMAGE", save, "images")
    
    # Validate
    errors = pipeline.validate()
    if errors:
        print(f"Pipeline validation failed:")
        for error in errors:
            print(f"  - {error}")
    else:
        print(f"Pipeline validation passed")
    
    # Display info
    print(f"\nPipeline Info:")
    print(f"  Name: {pipeline.name}")
    print(f"  Nodes: {len(pipeline.nodes)}")
    print(f"  Connections: {len(pipeline.connections)}")
    
    # Convert to dict
    pipeline_dict = pipeline.to_dict()
    print(f"\nNode List:")
    for node_id, node in pipeline_dict['nodes'].items():
        print(f"  {node_id}: {node['type']}")
    
    print("\nPipeline demo completed")


def demo_generation():
    """Generation demo (simulated)"""
    print("\n" + "=" * 60)
    print("Genesis Generation - Generation Demo")
    print("=" * 60)
    
    config = GenesisConfig(device='cuda')
    
    with GenesisEngine(config) as engine:
        # Simulated generation
        result = engine.generate(
            prompt="a beautiful landscape with mountains and lake",
            negative_prompt="ugly, blurry",
            width=512,
            height=512,
            steps=20,
            cfg_scale=7.0,
            seed=42
        )
        
        print(f"\nGeneration Result:")
        print(f"  Status: {result['status']}")
        print(f"  Success: {result['success']}")
        print(f"  Execution time: {result['execution_time']:.2f}s")
        if 'error' in result:
            print(f"  Error: {result['error']}")
    
    print("\nGeneration demo completed")


def main():
    """Main entry point with real acceleration"""
    parser = argparse.ArgumentParser(
        description='Genesis - Production AI Engine with PyTorch Acceleration'
    )
    parser.add_argument(
        '--mode',
        choices=['info', 'benchmark', 'mixed_precision', 'memory', 'all'],
        default='all',
        help='Demo mode to run'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    print("\n")
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║                                                                ║")
    print("║                    Genesis AI Engine                           ║")
    print("║                                                                ║")
    print("║          PyTorch Acceleration & Optimization Core              ║")
    print("║                                                                ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    try:
        if args.mode == 'info' or args.mode == 'all':
            show_system_info()
        
        if args.mode == 'benchmark' or args.mode == 'all':
            benchmark_acceleration()
        
        if args.mode == 'mixed_precision' or args.mode == 'all':
            demonstrate_mixed_precision()
        
        if args.mode == 'memory' or args.mode == 'all':
            show_memory_optimization()
        
        print("\n" + "="*70)
        print(" All demos completed successfully")
        print("="*70)
        print()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
