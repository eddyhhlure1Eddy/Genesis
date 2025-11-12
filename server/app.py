"""
Genesis Server Application
Main Flask application with dynamic node loading
Author: eddy
"""

import os
import logging
from pathlib import Path
from typing import Optional

try:
    from flask import Flask
    from flask_cors import CORS
    from flask_socketio import SocketIO
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Warning: Flask not installed. Install with: pip install flask flask-cors flask-socketio")

from ..core.config import GenesisConfig
from ..core.engine import GenesisEngine
from ..core.node_registry import get_node_registry
from ..core.node_scanner import get_node_scanner

logger = logging.getLogger(__name__)


class GenesisServer:
    """
    Genesis Server - Production-ready server with dynamic node loading

    Features:
    - Dynamic node loading (built-in + custom)
    - RESTful API
    - WebSocket support
    - Workflow execution
    - Progress tracking
    """

    def __init__(
        self,
        config: Optional[GenesisConfig] = None,
        host: str = "0.0.0.0",
        port: int = 5000,
        debug: bool = False
    ):
        """
        Initialize Genesis Server

        Args:
            config: Engine configuration
            host: Server host
            port: Server port
            debug: Debug mode
        """
        if not FLASK_AVAILABLE:
            raise RuntimeError("Flask is required. Install with: pip install flask flask-cors flask-socketio")

        self.config = config or GenesisConfig()
        self.host = host
        self.port = port
        self.debug = debug

        # Core components
        self.engine = GenesisEngine(self.config)
        self.registry = get_node_registry()
        self.scanner = get_node_scanner()

        # Flask app
        self.app = None
        self.socketio = None

        # Initialize
        self._initialized = False

        logger.info(f"Genesis Server initialized (host={host}, port={port})")

    def initialize(self):
        """Initialize server components"""
        if self._initialized:
            logger.warning("Server already initialized")
            return

        logger.info("Initializing Genesis Server...")

        # Initialize engine
        self.engine.initialize()

        # Load nodes
        self._load_all_nodes()

        # Create Flask app
        self.app = create_app(self)

        # Create SocketIO
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            async_mode='threading',
            logger=self.debug,
            engineio_logger=self.debug
        )

        # Register WebSocket handlers
        self._register_websocket_handlers()

        self._initialized = True
        logger.info("Genesis Server initialized successfully")

    def _load_all_nodes(self):
        """Load all nodes (built-in + custom)"""
        logger.info("Loading nodes...")

        # Discover and load all nodes
        stats = self.scanner.discover_all()

        logger.info(f"Node loading complete:")
        logger.info(f"  Built-in nodes: {stats['builtin']}")
        logger.info(f"  Custom nodes: {stats['custom']}")
        logger.info(f"  Total nodes: {stats['total']}")
        logger.info(f"  Failed modules: {stats['failed']}")

        if stats['failed'] > 0:
            logger.warning("Some modules failed to load. Check logs for details.")

        # Log node statistics
        reg_stats = self.registry.get_statistics()
        logger.info(f"Registry statistics:")
        logger.info(f"  Total nodes: {reg_stats['total_nodes']}")
        logger.info(f"  Categories: {reg_stats['total_categories']}")
        for category, count in reg_stats['nodes_by_category'].items():
            logger.info(f"    - {category}: {count} nodes")

    def _register_websocket_handlers(self):
        """Register WebSocket event handlers"""
        from .websocket.handlers import register_handlers
        register_handlers(self.socketio, self)

    def run(self):
        """Run the server"""
        if not self._initialized:
            self.initialize()

        logger.info(f"Starting Genesis Server on {self.host}:{self.port}")
        logger.info(f"Access the server at: http://localhost:{self.port}")
        logger.info(f"API documentation: http://localhost:{self.port}/api/docs")

        self.socketio.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=self.debug,
            use_reloader=False  # Disable reloader to prevent double initialization
        )

    def shutdown(self):
        """Shutdown server"""
        logger.info("Shutting down Genesis Server...")
        if self.engine:
            self.engine.cleanup()
        logger.info("Server shutdown complete")


def create_app(server: GenesisServer) -> Flask:
    """
    Create Flask application

    Args:
        server: GenesisServer instance

    Returns:
        Flask app
    """
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.urandom(24)
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload

    # Enable CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Store server instance
    app.genesis_server = server

    # Register blueprints
    from .routes import register_routes
    register_routes(app)

    # Basic routes
    @app.route('/')
    def index():
        return {
            'name': 'Genesis Server',
            'version': '0.1.0',
            'status': 'running',
            'nodes_loaded': len(server.registry),
            'api_docs': '/api/docs'
        }

    @app.route('/health')
    def health():
        return {'status': 'healthy', 'nodes': len(server.registry)}

    return app
