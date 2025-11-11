"""
Genesis Basic Usage Examples
Basic usage examples
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genesis import GenesisEngine
from genesis.core.config import GenesisConfig
from genesis.core.pipeline import Pipeline, PipelineBuilder


def example_1_basic_engine():
    """Example 1: Basic engine initialization"""
    print("\n" + "="*60)
    print("Example 1: Basic Engine Initialization")
    print("="*60)
    
    # Create config
    config = GenesisConfig(
        device='cuda',
        log_level='INFO'
    )
    
    # Create and initialize engine
    engine = GenesisEngine(config)
    engine.initialize()
    
    # Get device info
    device_info = engine.get_device_info()
    print("\nDevice Information:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    engine.cleanup()
    print("\n✓ Example 1 completed")


def example_2_context_manager():
    """Example 2: Using context manager"""
    print("\n" + "="*60)
    print("Example 2: Context Manager Usage")
    print("="*60)
    
    config = GenesisConfig(device='cuda')
    
    # Use with statement for automatic cleanup
    with GenesisEngine(config) as engine:
        print("\nEngine initialized automatically")
        print(f"Device: {engine.get_device_info()['device']}")
        
    print("\n✓ Example 2 completed (automatic cleanup)")


def example_3_list_models():
    """Example 3: List available models"""
    print("\n" + "="*60)
    print("Example 3: List Available Models")
    print("="*60)
    
    with GenesisEngine() as engine:
        try:
            models = engine.get_available_models()
            
            print("\nAvailable Models:")
            for model_type, model_list in models.items():
                print(f"\n{model_type.upper()}:")
                if model_list:
                    for model in model_list[:5]:  # Show first 5
                        print(f"  - {model}")
                    if len(model_list) > 5:
                        print(f"  ... and {len(model_list) - 5} more")
                else:
                    print("  (no models found)")
                    
        except Exception as e:
            print(f"\n⚠ Could not list models: {e}")
            print("  Make sure to place models in the correct directories")
    
    print("\n✓ Example 3 completed")


def example_4_simple_pipeline():
    """Example 4: Create a simple pipeline"""
    print("\n" + "="*60)
    print("Example 4: Simple Pipeline")
    print("="*60)
    
    # Create pipeline
    pipeline = Pipeline("simple_workflow")
    
    # Add nodes
    loader = pipeline.add_node("CheckpointLoader", checkpoint="model.safetensors")
    print(f"Added node: {loader} (CheckpointLoader)")
    
    sampler = pipeline.add_node("KSampler", steps=20, cfg=7.0)
    print(f"Added node: {sampler} (KSampler)")
    
    decoder = pipeline.add_node("VAEDecode")
    print(f"Added node: {decoder} (VAEDecode)")
    
    save = pipeline.add_node("SaveImage", filename_prefix="genesis")
    print(f"Added node: {save} (SaveImage)")
    
    # Connect nodes
    pipeline.connect(loader, "MODEL", sampler, "model")
    pipeline.connect(loader, "VAE", decoder, "vae")
    pipeline.connect(sampler, "LATENT", decoder, "samples")
    pipeline.connect(decoder, "IMAGE", save, "images")
    
    print(f"\nPipeline created:")
    print(f"  Nodes: {len(pipeline.nodes)}")
    print(f"  Connections: {len(pipeline.connections)}")
    
    # Validate
    errors = pipeline.validate()
    if errors:
        print(f"\n❌ Validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print(f"\n✓ Pipeline validation passed")
    
    print("\n✓ Example 4 completed")


def example_5_pipeline_builder():
    """Example 5: Use pipeline builder"""
    print("\n" + "="*60)
    print("Example 5: Pipeline Builder")
    print("="*60)
    
    # Use pre-built template
    pipeline = PipelineBuilder.text_to_image(
        checkpoint="model.safetensors",
        prompt="a beautiful landscape with mountains and lake",
        negative_prompt="ugly, blurry, low quality",
        width=512,
        height=512,
        steps=20,
        cfg=7.5,
        seed=42
    )
    
    print(f"\nPipeline created from template:")
    print(f"  Name: {pipeline.name}")
    print(f"  Nodes: {len(pipeline.nodes)}")
    print(f"  Connections: {len(pipeline.connections)}")
    
    # Show nodes
    print(f"\nNodes in pipeline:")
    for node_id, node in pipeline.nodes.items():
        print(f"  {node_id}: {node.type}")
    
    print("\n✓ Example 5 completed")


def example_6_generate_image():
    """Example 6: Generate image (simulated)"""
    print("\n" + "="*60)
    print("Example 6: Image Generation")
    print("="*60)
    
    config = GenesisConfig(
        device='cuda',
        output_dir=Path('output')
    )
    
    with GenesisEngine(config) as engine:
        # Generate image
        result = engine.generate(
            prompt="a serene mountain landscape at sunset",
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
        print(f"  Time: {result['execution_time']:.2f}s")
        print(f"  Prompt: {result['prompt']}")
        
        if not result['success']:
            print(f"  Error: {result.get('error', 'Unknown error')}")
    
    print("\n✓ Example 6 completed")


def example_7_custom_config():
    """Example 7: Custom configuration"""
    print("\n" + "="*60)
    print("Example 7: Custom Configuration")
    print("="*60)
    
    # Create custom config
    config = GenesisConfig(
        device='cuda',
        vram_mode='high',
        checkpoints_dir=Path('models/checkpoints'),
        output_dir=Path('output/custom'),
        log_level='DEBUG',
        num_threads=8,
        enable_cache=True,
        cache_size_mb=4096
    )
    
    print("\nCustom Configuration:")
    config_dict = config.to_dict()
    for key, value in config_dict.items():
        print(f"  {key}: {value}")
    
    # Create directories
    config.create_directories()
    print("\n✓ Directories created")
    
    print("\n✓ Example 7 completed")


def main():
    """Run all examples"""
    print("\n")
    print("╔════════════════════════════════════════════════════════╗")
    print("║                                                        ║")
    print("║           Genesis Usage Examples                       ║")
    print("║                                                        ║")
    print("╚════════════════════════════════════════════════════════╝")
    
    examples = [
        ("Basic Engine", example_1_basic_engine),
        ("Context Manager", example_2_context_manager),
        ("List Models", example_3_list_models),
        ("Simple Pipeline", example_4_simple_pipeline),
        ("Pipeline Builder", example_5_pipeline_builder),
        ("Generate Image", example_6_generate_image),
        ("Custom Config", example_7_custom_config),
    ]
    
    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all examples...")
    
    try:
        for name, example_func in examples:
            example_func()
            
        print("\n" + "="*60)
        print("✓ All examples completed successfully!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
