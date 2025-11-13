"""
Genesis AI Engine

Lightweight Diffusion Engine built from scratch
Focus on simplicity, efficiency, and ease of use

Author: eddy
Date: 2025-11-12
"""

__version__ = "0.1.0"
__author__ = "eddy"

# Core imports (lightweight - no node system loaded)
from .core.engine import GenesisEngine
from .core.genesis_core import GenesisCore
from .core.pipeline import Pipeline, PipelineBuilder
from .core.config import GenesisConfig
from .core.optimization import CoreOptimizer, create_optimized_device
from .models.loader import ModelLoader


def get_node_registry():
    """
    Get NODE_REGISTRY (lazy import)

    Call this function only when you need node functionality.
    This prevents automatic node loading on startup.

    Returns:
        NodeRegistry: Global node registry
    """
    from .core.nodes import NODE_REGISTRY
    return NODE_REGISTRY


def get_base_node():
    """
    Get BaseNode class (lazy import)

    Returns:
        BaseNode: Base node class
    """
    from .core.nodes import BaseNode
    return BaseNode


def register_wanvideo_nodes():
    """
    Register WanVideo nodes into the node registry

    Returns:
        bool: True if successful
    """
    from .nodes import wanvideo_nodes
    return True


# Default exports (no NODE_REGISTRY to prevent auto-loading)
__all__ = [
    'GenesisEngine',
    'GenesisCore',
    'Pipeline',
    'PipelineBuilder',
    'GenesisConfig',
    'ModelLoader',
    'CoreOptimizer',
    'create_optimized_device',
    'get_node_registry',
    'get_base_node',
    'register_wanvideo_nodes',
]
