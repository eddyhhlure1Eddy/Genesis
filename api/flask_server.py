"""
Genesis Flask API Server
RESTful API using Flask framework
Author: eddy
"""

import os
import json
import base64
from io import BytesIO
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from flask import Flask, request, jsonify, send_file
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Warning: Flask not installed. Install with: pip install flask flask-cors")

from ..core.engine import GenesisEngine
from ..core.config import GenesisConfig
from ..core.pipeline import Pipeline


class GenesisFlaskAPI:
    """
    Genesis Flask API Server
    
    Provides RESTful API for Genesis engine using Flask
    """
    
    def __init__(
        self,
        engine: Optional[GenesisEngine] = None,
        host: str = "0.0.0.0",
        port: int = 5000,
        debug: bool = False
    ):
        """
        Initialize Flask API
        
        Args:
            engine: Genesis engine instance
            host: Server host
            port: Server port
            debug: Debug mode
        """
        if not FLASK_AVAILABLE:
            raise RuntimeError("Flask is required. Install with: pip install flask flask-cors")
        
        self.engine = engine or GenesisEngine()
        self.host = host
        self.port = port
        self.debug = debug
        
        # Create Flask app
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS
        
        # Setup routes
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.route('/', methods=['GET'])
        def index():
            """API index"""
            return jsonify({
                'name': 'Genesis API',
                'version': '0.1.0',
                'description': 'RESTful API for Genesis AI Engine',
                'endpoints': {
                    'GET /': 'API information',
                    'GET /health': 'Health check',
                    'GET /info': 'Engine information',
                    'GET /models': 'List available models',
                    'GET /models/checkpoints': 'List checkpoints',
                    'GET /models/vae': 'List VAE models',
                    'GET /models/loras': 'List LoRA models',
                    'GET /device': 'Device information',
                    'GET /samplers': 'List available samplers',
                    'POST /generate': 'Generate image',
                    'POST /generate/batch': 'Batch generate images',
                    'POST /pipeline/execute': 'Execute pipeline',
                    'POST /pipeline/validate': 'Validate pipeline',
                    'POST /model/load': 'Load model',
                    'POST /model/unload': 'Unload model',
                }
            })
        
        @self.app.route('/health', methods=['GET'])
        def health():
            """Health check"""
            return jsonify({
                'status': 'healthy',
                'initialized': self.engine._initialized
            })
        
        @self.app.route('/info', methods=['GET'])
        def info():
            """Engine information"""
            return jsonify({
                'name': 'Genesis',
                'version': '0.1.0',
                'author': 'eddy',
                'initialized': self.engine._initialized,
                'device': str(self.engine._device) if self.engine._device else None
            })
        
        @self.app.route('/models', methods=['GET'])
        def list_models():
            """List all available models"""
            try:
                if not self.engine._initialized:
                    self.engine.initialize()
                
                models = self.engine.get_available_models()
                return jsonify({
                    'success': True,
                    'models': models
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/models/checkpoints', methods=['GET'])
        def list_checkpoints():
            """List checkpoint models"""
            try:
                if not self.engine._initialized:
                    self.engine.initialize()
                
                checkpoints = self.engine.model_loader.list_checkpoints()
                return jsonify({
                    'success': True,
                    'checkpoints': checkpoints,
                    'count': len(checkpoints)
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/models/vae', methods=['GET'])
        def list_vae():
            """List VAE models"""
            try:
                if not self.engine._initialized:
                    self.engine.initialize()
                
                vae_models = self.engine.model_loader.list_vae()
                return jsonify({
                    'success': True,
                    'vae': vae_models,
                    'count': len(vae_models)
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/models/loras', methods=['GET'])
        def list_loras():
            """List LoRA models"""
            try:
                if not self.engine._initialized:
                    self.engine.initialize()
                
                loras = self.engine.model_loader.list_loras()
                return jsonify({
                    'success': True,
                    'loras': loras,
                    'count': len(loras)
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/device', methods=['GET'])
        def device_info():
            """Device information"""
            try:
                if not self.engine._initialized:
                    self.engine.initialize()
                
                info = self.engine.get_device_info()
                return jsonify({
                    'success': True,
                    'device': info
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/samplers', methods=['GET'])
        def list_samplers():
            """List available samplers and schedulers"""
            from ..core.samplers import SamplerRegistry
            
            return jsonify({
                'success': True,
                'samplers': SamplerRegistry.get_sampler_names(),
                'schedulers': SamplerRegistry.get_scheduler_names()
            })
        
        @self.app.route('/generate', methods=['POST'])
        def generate():
            """Generate image"""
            try:
                data = request.get_json()
                
                if not data:
                    return jsonify({
                        'success': False,
                        'error': 'No data provided'
                    }), 400
                
                # Extract parameters
                prompt = data.get('prompt', '')
                negative_prompt = data.get('negative_prompt', '')
                width = data.get('width', 512)
                height = data.get('height', 512)
                steps = data.get('steps', 20)
                cfg_scale = data.get('cfg_scale', 7.0)
                seed = data.get('seed')
                sampler = data.get('sampler', 'euler')
                scheduler = data.get('scheduler', 'normal')
                
                # Initialize engine
                if not self.engine._initialized:
                    self.engine.initialize()
                
                # Generate
                result = self.engine.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    seed=seed
                )
                
                return jsonify(result)
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/generate/batch', methods=['POST'])
        def generate_batch():
            """Batch generate images"""
            try:
                data = request.get_json()
                
                if not data or 'requests' not in data:
                    return jsonify({
                        'success': False,
                        'error': 'No requests provided'
                    }), 400
                
                requests_list = data['requests']
                results = []
                
                # Initialize engine
                if not self.engine._initialized:
                    self.engine.initialize()
                
                # Process each request
                for req in requests_list:
                    try:
                        result = self.engine.generate(**req)
                        results.append(result)
                    except Exception as e:
                        results.append({
                            'success': False,
                            'error': str(e)
                        })
                
                return jsonify({
                    'success': True,
                    'results': results,
                    'count': len(results)
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/pipeline/execute', methods=['POST'])
        def execute_pipeline():
            """Execute pipeline"""
            try:
                data = request.get_json()
                
                if not data:
                    return jsonify({
                        'success': False,
                        'error': 'No pipeline data provided'
                    }), 400
                
                # Create pipeline from data
                pipeline = Pipeline.from_dict(data)
                
                # Initialize engine
                if not self.engine._initialized:
                    self.engine.initialize()
                
                # Execute
                result = self.engine.executor.execute_pipeline(pipeline)
                
                return jsonify(result)
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/pipeline/validate', methods=['POST'])
        def validate_pipeline():
            """Validate pipeline"""
            try:
                data = request.get_json()
                
                if not data:
                    return jsonify({
                        'success': False,
                        'error': 'No pipeline data provided'
                    }), 400
                
                # Create pipeline
                pipeline = Pipeline.from_dict(data)
                
                # Validate
                errors = pipeline.validate()
                
                return jsonify({
                    'success': len(errors) == 0,
                    'valid': len(errors) == 0,
                    'errors': errors
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/model/load', methods=['POST'])
        def load_model():
            """Load model"""
            try:
                data = request.get_json()
                
                checkpoint = data.get('checkpoint')
                vae = data.get('vae')
                
                if not checkpoint:
                    return jsonify({
                        'success': False,
                        'error': 'No checkpoint specified'
                    }), 400
                
                # Initialize engine
                if not self.engine._initialized:
                    self.engine.initialize()
                
                # Load model
                result = self.engine.load_model(checkpoint, vae)
                
                return jsonify({
                    'success': True,
                    'model': result
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/model/unload', methods=['POST'])
        def unload_model():
            """Unload model"""
            try:
                data = request.get_json()
                model_name = data.get('model_name')
                
                if not model_name:
                    return jsonify({
                        'success': False,
                        'error': 'No model name specified'
                    }), 400
                
                # Unload
                self.engine.model_loader.unload_model(model_name)
                
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
        """Start Flask server"""
        print(f"Starting Genesis Flask API on http://{self.host}:{self.port}")
        self.app.run(
            host=self.host,
            port=self.port,
            debug=self.debug
        )


def create_flask_api(
    config: Optional[GenesisConfig] = None,
    host: str = "0.0.0.0",
    port: int = 5000,
    debug: bool = False
) -> GenesisFlaskAPI:
    """
    Create Flask API server
    
    Args:
        config: Engine configuration
        host: Server host
        port: Server port
        debug: Debug mode
        
    Returns:
        GenesisFlaskAPI instance
    """
    engine = GenesisEngine(config or GenesisConfig())
    api = GenesisFlaskAPI(engine, host, port, debug)
    return api


if __name__ == "__main__":
    # Example usage
    api = create_flask_api(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
    api.run()
