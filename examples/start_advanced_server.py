"""
Start Genesis Advanced Server
Launch script for the advanced server with WebSocket support
Author: eddy
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from genesis.api import create_advanced_server
from genesis.core.config import GenesisConfig


def main():
    print("="*60)
    print("Genesis Advanced Server Launcher")
    print("="*60)
    print()
    
    # Configuration
    config = GenesisConfig(
        device='cuda',
        log_level='INFO',
        checkpoints_dir=Path('models/checkpoints'),
        vae_dir=Path('models/vae'),
        lora_dir=Path('models/loras'),
        output_dir=Path('output')
    )
    
    # Server settings
    HOST = "0.0.0.0"
    PORT = 5000
    DEBUG = True
    
    print("Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Host: {HOST}")
    print(f"  Port: {PORT}")
    print(f"  Debug: {DEBUG}")
    print()
    
    print("Features:")
    print("  ✓ RESTful HTTP API")
    print("  ✓ WebSocket real-time communication")
    print("  ✓ Task queue with progress tracking")
    print("  ✓ Session management")
    print("  ✓ Multi-client support")
    print()
    
    print("Supported Clients:")
    print("  - Web browsers (JavaScript/WebSocket)")
    print("  - Tkinter applications")
    print("  - PyQt/PySide applications")
    print("  - Command-line scripts")
    print("  - Any HTTP/WebSocket client")
    print()
    
    print("API Endpoints:")
    print("  HTTP:")
    print("    GET  /              - API information")
    print("    GET  /health        - Health check")
    print("    POST /api/session/create")
    print("    POST /api/task/submit")
    print("    GET  /api/task/<id>")
    print("    POST /api/task/<id>/cancel")
    print("    GET  /api/tasks")
    print("    GET  /api/models")
    print("    GET  /api/device")
    print()
    print("  WebSocket:")
    print("    connect, disconnect")
    print("    join, leave (rooms)")
    print("    submit_task")
    print("    get_task_status")
    print("    progress (server -> client)")
    print("    task_complete (server -> client)")
    print("    task_error (server -> client)")
    print()
    
    # Create server
    server = create_advanced_server(
        config=config,
        host=HOST,
        port=PORT,
        debug=DEBUG
    )
    
    print("Starting server...")
    print(f"Access at: http://{HOST}:{PORT}")
    print()
    print("Press Ctrl+C to stop")
    print("="*60)
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        print("Goodbye!")


if __name__ == "__main__":
    main()
