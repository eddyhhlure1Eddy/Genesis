"""
Genesis API Client Example
Example of using Genesis API with requests
Author: eddy
"""

import requests
import json


class GenesisAPIClient:
    """Simple client for Genesis API"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """
        Initialize API client
        
        Args:
            base_url: Base URL of Genesis API
        """
        self.base_url = base_url.rstrip('/')
    
    def get(self, endpoint: str):
        """GET request"""
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url)
        return response.json()
    
    def post(self, endpoint: str, data: dict):
        """POST request"""
        url = f"{self.base_url}{endpoint}"
        response = requests.post(url, json=data)
        return response.json()
    
    def health_check(self):
        """Check API health"""
        return self.get('/health')
    
    def get_info(self):
        """Get engine info"""
        return self.get('/info')
    
    def list_models(self):
        """List all models"""
        return self.get('/models')
    
    def list_checkpoints(self):
        """List checkpoint models"""
        return self.get('/models/checkpoints')
    
    def list_samplers(self):
        """List available samplers"""
        return self.get('/samplers')
    
    def get_device_info(self):
        """Get device information"""
        return self.get('/device')
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: int = None,
        sampler: str = "euler",
        scheduler: str = "normal"
    ):
        """
        Generate image
        
        Args:
            prompt: Positive prompt
            negative_prompt: Negative prompt
            width: Image width
            height: Image height
            steps: Sampling steps
            cfg_scale: CFG scale
            seed: Random seed
            sampler: Sampler name
            scheduler: Scheduler name
            
        Returns:
            Generation result
        """
        data = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'width': width,
            'height': height,
            'steps': steps,
            'cfg_scale': cfg_scale,
            'seed': seed,
            'sampler': sampler,
            'scheduler': scheduler
        }
        
        return self.post('/generate', data)
    
    def generate_batch(self, requests_list: list):
        """
        Batch generate images
        
        Args:
            requests_list: List of generation requests
            
        Returns:
            Batch results
        """
        data = {'requests': requests_list}
        return self.post('/generate/batch', data)
    
    def load_model(self, checkpoint: str, vae: str = None):
        """
        Load model
        
        Args:
            checkpoint: Checkpoint filename
            vae: VAE filename (optional)
            
        Returns:
            Load result
        """
        data = {
            'checkpoint': checkpoint,
            'vae': vae
        }
        return self.post('/model/load', data)


def example_1_basic():
    """Example 1: Basic API usage"""
    print("\n" + "=" * 60)
    print("Example 1: Basic API Usage")
    print("=" * 60)
    
    client = GenesisAPIClient("http://localhost:5000")
    
    # Health check
    health = client.health_check()
    print(f"\nHealth: {health}")
    
    # Engine info
    info = client.get_info()
    print(f"\nEngine Info:")
    print(f"  Name: {info.get('name')}")
    print(f"  Version: {info.get('version')}")
    print(f"  Initialized: {info.get('initialized')}")
    
    # Device info
    device = client.get_device_info()
    print(f"\nDevice: {device}")


def example_2_list_resources():
    """Example 2: List available resources"""
    print("\n" + "=" * 60)
    print("Example 2: List Resources")
    print("=" * 60)
    
    client = GenesisAPIClient("http://localhost:5000")
    
    # List models
    models = client.list_models()
    print(f"\nAvailable Models:")
    if models.get('success'):
        for model_type, model_list in models['models'].items():
            print(f"  {model_type}: {len(model_list)} models")
    
    # List samplers
    samplers = client.list_samplers()
    print(f"\nAvailable Samplers:")
    if samplers.get('success'):
        print(f"  Samplers: {', '.join(samplers['samplers'][:5])}...")
        print(f"  Schedulers: {', '.join(samplers['schedulers'])}")


def example_3_generate():
    """Example 3: Generate image"""
    print("\n" + "=" * 60)
    print("Example 3: Generate Image")
    print("=" * 60)
    
    client = GenesisAPIClient("http://localhost:5000")
    
    # Generate
    result = client.generate(
        prompt="a beautiful mountain landscape at sunset",
        negative_prompt="ugly, blurry, low quality",
        width=512,
        height=512,
        steps=20,
        cfg_scale=7.5,
        seed=42
    )
    
    print(f"\nGeneration Result:")
    print(f"  Success: {result.get('success')}")
    print(f"  Status: {result.get('status')}")
    print(f"  Time: {result.get('execution_time', 0):.2f}s")
    
    if not result.get('success'):
        print(f"  Error: {result.get('error')}")


def example_4_batch_generate():
    """Example 4: Batch generate"""
    print("\n" + "=" * 60)
    print("Example 4: Batch Generate")
    print("=" * 60)
    
    client = GenesisAPIClient("http://localhost:5000")
    
    # Batch requests
    requests_list = [
        {
            'prompt': 'a cat',
            'width': 512,
            'height': 512,
            'steps': 15
        },
        {
            'prompt': 'a dog',
            'width': 512,
            'height': 512,
            'steps': 15
        },
        {
            'prompt': 'a bird',
            'width': 512,
            'height': 512,
            'steps': 15
        }
    ]
    
    result = client.generate_batch(requests_list)
    
    print(f"\nBatch Generation Result:")
    print(f"  Success: {result.get('success')}")
    print(f"  Count: {result.get('count')}")
    
    if result.get('success'):
        for i, res in enumerate(result['results'], 1):
            print(f"  Image {i}: {res.get('status')}")


def main():
    """Run all examples"""
    print("\n")
    print("╔════════════════════════════════════════════════════════╗")
    print("║                                                        ║")
    print("║         Genesis API Client Examples                   ║")
    print("║                                                        ║")
    print("╚════════════════════════════════════════════════════════╝")
    
    print("\nMake sure Genesis Flask API is running on http://localhost:5000")
    print("Run: python examples/flask_api_example.py")
    
    input("\nPress Enter to continue...")
    
    try:
        example_1_basic()
        example_2_list_resources()
        example_3_generate()
        example_4_batch_generate()
        
        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server")
        print("Make sure the Flask API is running on http://localhost:5000")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
