"""
ComfyUI Workflow Conversion Example
Demonstrate ComfyUI workflow parsing and execution
Author: eddy
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from genesis.core import (
    ComfyUIWorkflowConverter,
    load_and_execute_workflow,
    register_comfyui_node,
    ComfyUINodeInterface
)


@register_comfyui_node("ExampleTextEncode")
class ExampleTextEncodeNode(ComfyUINodeInterface):
    """Example text encoding node"""
    
    CATEGORY = "conditioning"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ("CONDITIONING",)
    
    def execute(self, text):
        print(f"Text Encode: {text}")
        conditioning = {"text": text, "embeddings": [0.1, 0.2, 0.3]}
        return (conditioning,)


@register_comfyui_node("ExampleSampler")
class ExampleSamplerNode(ComfyUINodeInterface):
    """Example sampler node"""
    
    CATEGORY = "sampling"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ("LATENT",)
    
    def execute(self, positive, negative, steps):
        print(f"Sampler: steps={steps}")
        print(f"  Positive: {positive.get('text')}")
        print(f"  Negative: {negative.get('text')}")
        latent = {"samples": [[1, 2, 3], [4, 5, 6]]}
        return (latent,)


@register_comfyui_node("ExampleDecode")
class ExampleDecodeNode(ComfyUINodeInterface):
    """Example decode node"""
    
    CATEGORY = "image"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
            }
        }
    
    @classmethod
    def RETURN_TYPES(cls):
        return ("IMAGE",)
    
    def execute(self, samples):
        print(f"Decode: {samples}")
        image = {"pixels": "fake_image_data"}
        return (image,)


def create_example_workflow():
    """Create example ComfyUI workflow"""
    workflow = {
        "1": {
            "class_type": "ExampleTextEncode",
            "inputs": {
                "text": "beautiful landscape"
            }
        },
        "2": {
            "class_type": "ExampleTextEncode",
            "inputs": {
                "text": "ugly, blurry"
            }
        },
        "3": {
            "class_type": "ExampleSampler",
            "inputs": {
                "positive": ["1", 0],
                "negative": ["2", 0],
                "steps": 20
            }
        },
        "4": {
            "class_type": "ExampleDecode",
            "inputs": {
                "samples": ["3", 0]
            }
        }
    }
    
    return workflow


def demo_workflow_conversion():
    """Demonstrate workflow conversion"""
    print("="*70)
    print(" ComfyUI Workflow Conversion Demo")
    print("="*70)
    
    workflow = create_example_workflow()
    
    print("\nOriginal ComfyUI Workflow:")
    print(json.dumps(workflow, indent=2))
    
    converter = ComfyUIWorkflowConverter()
    
    print("\nParsing workflow...")
    if converter.parse_workflow(workflow):
        print("Workflow parsed successfully!")
        
        info = converter.get_workflow_info()
        print(f"\nWorkflow Info:")
        print(f"  Node count: {info['node_count']}")
        print(f"  Execution order: {' -> '.join(info['execution_order'])}")
        
        print("\nConverting to Genesis format...")
        genesis_result = converter.convert_to_genesis(workflow)
        
        if genesis_result['success']:
            print("Conversion successful!")
            print("\nGenesis Workflow:")
            print(json.dumps(genesis_result['workflow'], indent=2))
        
        print("\n" + "="*70)
        print(" Executing Workflow")
        print("="*70)
        
        result = converter.execute_workflow()
        
        if result['success']:
            print("\nWorkflow executed successfully!")
            print(f"\nResults:")
            for node_id, node_result in result['results'].items():
                status = "SUCCESS" if node_result['success'] else "FAILED"
                print(f"  Node {node_id} ({node_result['class_type']}): {status}")
                if not node_result['success']:
                    print(f"    Error: {node_result.get('error')}")
        else:
            print(f"\nWorkflow execution failed: {result.get('error')}")
    
    else:
        print("Failed to parse workflow")
    
    print("\n" + "="*70)


def demo_load_from_file():
    """Demonstrate loading workflow from file"""
    print("\n" + "="*70)
    print(" Load Workflow from File Demo")
    print("="*70)
    
    workflow = create_example_workflow()
    
    workflow_file = Path("example_workflow.json")
    with open(workflow_file, 'w', encoding='utf-8') as f:
        json.dump(workflow, f, indent=2)
    
    print(f"\nSaved workflow to: {workflow_file}")
    
    print("\nLoading and executing workflow from file...")
    result = load_and_execute_workflow(str(workflow_file))
    
    if result['success']:
        print("Workflow executed successfully!")
        print(f"Processed {len(result['results'])} nodes")
    else:
        print(f"Execution failed: {result.get('error')}")
    
    workflow_file.unlink()
    print(f"\nCleaned up: {workflow_file}")
    
    print("\n" + "="*70)


def main():
    """Run all demos"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + " ComfyUI Workflow Conversion - Genesis Integration ".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    demo_workflow_conversion()
    demo_load_from_file()
    
    print("\n" + "="*70)
    print(" All demos completed")
    print("="*70)
    print()


if __name__ == "__main__":
    main()
