"""
Genesis Server - System API Routes
API endpoints for system information
Author: eddy
"""

from flask import Blueprint, jsonify, current_app
import logging
import torch
import sys
import platform

logger = logging.getLogger(__name__)

system_bp = Blueprint('system', __name__)


@system_bp.route('/info', methods=['GET'])
def get_system_info():
    """Get system information"""
    try:
        server = current_app.genesis_server

        info = {
            'genesis': {
                'version': '0.1.0',
                'nodes_loaded': len(server.registry),
                'categories': len(server.registry.get_categories())
            },
            'python': {
                'version': sys.version,
                'platform': platform.platform()
            },
            'pytorch': {
                'version': torch.__version__,
                'cuda_available': torch.cuda.is_available()
            }
        }

        # Add CUDA info if available
        if torch.cuda.is_available():
            info['cuda'] = {
                'version': torch.version.cuda,
                'device_count': torch.cuda.device_count(),
                'devices': []
            }

            for i in range(torch.cuda.device_count()):
                device_info = {
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'total_memory': torch.cuda.get_device_properties(i).total_memory / (1024**3),
                    'allocated_memory': torch.cuda.memory_allocated(i) / (1024**3),
                    'reserved_memory': torch.cuda.memory_reserved(i) / (1024**3)
                }
                info['cuda']['devices'].append(device_info)

        return jsonify({
            'success': True,
            'system': info
        })

    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@system_bp.route('/models', methods=['GET'])
def list_models():
    """Get list of available models"""
    try:
        server = current_app.genesis_server
        engine = server.engine

        models = engine.get_available_models()

        return jsonify({
            'success': True,
            'models': models
        })

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@system_bp.route('/device', methods=['GET'])
def get_device_info():
    """Get device information"""
    try:
        server = current_app.genesis_server
        engine = server.engine

        device_info = engine.get_device_info()

        return jsonify({
            'success': True,
            'device': device_info
        })

    except Exception as e:
        logger.error(f"Failed to get device info: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@system_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        server = current_app.genesis_server

        health = {
            'status': 'healthy',
            'nodes_loaded': len(server.registry),
            'engine_initialized': server.engine._initialized
        }

        return jsonify({
            'success': True,
            'health': health
        })

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'health': {'status': 'unhealthy'}
        }), 500
