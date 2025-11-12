"""
Genesis REST API Interface
Simple REST API adapter for Genesis Core
Author: eddy
"""

import logging
from typing import Optional
from flask import Flask, request, jsonify
from flask_cors import CORS

logger = logging.getLogger(__name__)


class GenesisRESTAPI:
    """
    REST API Interface for Genesis Core

    Features:
    - Simple HTTP API
    - No UI dependencies
    - Pure adapter
    """

    def __init__(self, core, host: str = "0.0.0.0", port: int = 5000):
        """
        Initialize REST API

        Args:
            core: Genesis Core instance
            host: Server host
            port: Server port
        """
        self.core = core
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        CORS(self.app)

        self._register_routes()

        logger.info(f"REST API initialized (host={host}, port={port})")

    def _register_routes(self):
        """Register API routes"""

        @self.app.route('/')
        def index():
            return {
                'name': 'Genesis Core API',
                'version': '0.1.0',
                'status': 'running',
                'endpoints': {
                    'execute': '/api/execute',
                    'workflow': '/api/workflow',
                    'status': '/api/status',
                    'health': '/health'
                }
            }

        @self.app.route('/health')
        def health():
            return {'status': 'healthy'}

        @self.app.route('/api/status')
        def get_status():
            try:
                status = self.core.get_status()
                return jsonify({
                    'success': True,
                    'status': status
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        @self.app.route('/api/execute', methods=['POST'])
        def execute():
            try:
                task = request.get_json()

                if not task:
                    return jsonify({
                        'success': False,
                        'error': 'Task data is required'
                    }), 400

                result = self.core.execute(task)

                return jsonify(result)

            except Exception as e:
                logger.error(f"Execution error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        @self.app.route('/api/workflow', methods=['POST'])
        def execute_workflow():
            try:
                data = request.get_json()

                if not data or 'workflow' not in data:
                    return jsonify({
                        'success': False,
                        'error': 'Workflow data is required'
                    }), 400

                workflow = data['workflow']
                result = self.core.execute_workflow(workflow)

                return jsonify(result)

            except Exception as e:
                logger.error(f"Workflow execution error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        @self.app.route('/api/models', methods=['GET'])
        def list_models():
            try:
                models = list(self.core.models.keys())
                return jsonify({
                    'success': True,
                    'models': models,
                    'total': len(models)
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        @self.app.route('/api/models/load', methods=['POST'])
        def load_model():
            try:
                data = request.get_json()
                model_name = data.get('model_name')

                if not model_name:
                    return jsonify({
                        'success': False,
                        'error': 'model_name is required'
                    }), 400

                model = self.core.load_model(model_name)

                return jsonify({
                    'success': True,
                    'model': model
                })

            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        @self.app.route('/api/models/unload', methods=['POST'])
        def unload_model():
            try:
                data = request.get_json()
                model_name = data.get('model_name')

                if not model_name:
                    return jsonify({
                        'success': False,
                        'error': 'model_name is required'
                    }), 400

                self.core.unload_model(model_name)

                return jsonify({
                    'success': True,
                    'message': f'Model {model_name} unloaded'
                })

            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

    def run(self):
        """Run the API server"""
        logger.info(f"Starting REST API server on {self.host}:{self.port}")

        print(f"\n{'='*70}")
        print(f" Genesis Core API Server")
        print(f"{'='*70}")
        print(f"\n  Status: Running")
        print(f"  URL: http://localhost:{self.port}")
        print(f"  Device: {self.core.device}")
        print(f"\n  Endpoints:")
        print(f"    - GET  /              # API info")
        print(f"    - GET  /health        # Health check")
        print(f"    - GET  /api/status    # Core status")
        print(f"    - POST /api/execute   # Execute task")
        print(f"    - POST /api/workflow  # Execute workflow")
        print(f"    - GET  /api/models    # List models")
        print(f"\n  Press Ctrl+C to stop")
        print(f"{'='*70}\n")

        self.app.run(
            host=self.host,
            port=self.port,
            debug=False,
            threaded=True
        )

    def start(self):
        """Alias for run()"""
        self.run()
