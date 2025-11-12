"""
Custom Node Loader for Genesis
Loads external ComfyUI-compatible custom nodes
Author: eddy
"""

import sys
import importlib.util
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class CustomNodeLoader:
    """Loads custom nodes from external directories"""
    
    def __init__(self):
        self.loaded_nodes = {}
        self.custom_node_paths = []
    
    def add_custom_node_path(self, path: str):
        """Add a custom node directory to search path"""
        node_path = Path(path)
        if node_path.exists() and node_path.is_dir():
            self.custom_node_paths.append(node_path)
            logger.info(f"Added custom node path: {node_path}")
        else:
            logger.warning(f"Custom node path not found: {node_path}")
    
    def load_custom_nodes(self):
        """Load all custom nodes from registered paths"""
        for path in self.custom_node_paths:
            self._load_nodes_from_directory(path)
    
    def scan_and_load_custom_nodes(self, base_directory: str):
        """Scan custom_nodes directory and load all subdirectories"""
        base_path = Path(base_directory)
        if not base_path.exists():
            logger.warning(f"Custom nodes directory not found: {base_path}")
            return
        
        logger.info(f"Scanning for custom nodes in: {base_path}")
        
        # Load ComfyUI compatibility layer before loading custom nodes
        self._ensure_comfy_compatibility()
        
        # Scan all subdirectories
        for subdir in base_path.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('.') and not subdir.name.startswith('__'):
                init_file = subdir / "__init__.py"
                if init_file.exists():
                    logger.info(f"Found custom node package: {subdir.name}")
                    self.add_custom_node_path(str(subdir))
        
        # Load all discovered nodes
        self.load_custom_nodes()
    
    def _ensure_comfy_compatibility(self):
        """Load ComfyUI compatibility layer if not already loaded"""
        if 'comfy' not in sys.modules:
            try:
                from genesis.compat import comfy_complete
                logger.info("ComfyUI compatibility layer loaded for custom nodes")
            except Exception as e:
                logger.error(f"Failed to load ComfyUI compatibility layer: {e}")
    
    def _load_nodes_from_directory(self, directory: Path):
        """Load nodes from a specific directory"""
        logger.info(f"Loading custom nodes from: {directory}")
        
        # Add to Python path
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
        
        # Look for __init__.py
        init_file = directory / "__init__.py"
        if init_file.exists():
            try:
                module_name = f"genesis_custom_{directory.name}"
                spec = importlib.util.spec_from_file_location(module_name, init_file)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                
                # Set __package__ to avoid relative import errors
                module.__package__ = module_name
                
                spec.loader.exec_module(module)
                
                # Get NODE_CLASS_MAPPINGS if available
                if hasattr(module, 'NODE_CLASS_MAPPINGS'):
                    mappings = module.NODE_CLASS_MAPPINGS
                    self.loaded_nodes.update(mappings)
                    logger.info(f"Loaded {len(mappings)} nodes from {directory.name}")
                else:
                    logger.warning(f"No NODE_CLASS_MAPPINGS found in {directory.name}")
                
            except Exception as e:
                logger.error(f"Failed to load custom nodes from {directory}: {e}")
    
    def get_node_class(self, node_type: str):
        """Get a custom node class by type"""
        return self.loaded_nodes.get(node_type)
    
    def get_all_nodes(self) -> Dict[str, Any]:
        """Get all loaded custom nodes"""
        return self.loaded_nodes.copy()


# Global custom node loader instance
custom_node_loader = CustomNodeLoader()
