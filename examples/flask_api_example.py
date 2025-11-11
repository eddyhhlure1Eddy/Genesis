"""
Genesis Flask API Example
Example of running Genesis with Flask API
Author: eddy
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genesis.api import create_flask_api
from genesis.core.config import GenesisConfig


def main():
    """Run Flask API server"""
    
    print("=" * 60)
    print("Genesis Flask API Server")
    print("=" * 60)
    
    # Create configuration
    config = GenesisConfig(
        device='cuda',
        log_level='INFO',
        checkpoints_dir=Path('models/checkpoints'),
        output_dir=Path('output')
    )
    
    # Create Flask API
    api = create_flask_api(
        config=config,
        host="0.0.0.0",
        port=5000,
        debug=True
    )
    
    print("\nAPI Endpoints:")
    print("  GET  /              - API information")
    print("  GET  /health        - Health check")
    print("  GET  /info          - Engine info")
    print("  GET  /models        - List all models")
    print("  GET  /device        - Device info")
    print("  GET  /samplers      - List samplers")
    print("  POST /generate      - Generate image")
    print("  POST /generate/batch - Batch generate")
    print("  POST /pipeline/execute - Execute pipeline")
    print()
    
    # Run server
    try:
        api.run()
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")


if __name__ == "__main__":
    main()
