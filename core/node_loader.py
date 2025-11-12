"""
Dynamic Node Loader for Genesis
Load node modules on-demand, avoiding global imports
Author: eddy
"""

import importlib
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class DynamicNodeLoader:
    """Dynamic node loader - loads node modules on demand"""

    # Available built-in node modules
    BUILTIN_MODULES = {
        'loaders': 'genesis.nodes.loaders',
        'samplers': 'genesis.nodes.samplers',
        'conditioning': 'genesis.nodes.conditioning',
        'latent': 'genesis.nodes.latent',
        'image': 'genesis.nodes.image',
        'wan_video': 'genesis.nodes.wan_video',
    }

    def __init__(self):
        self.loaded_modules = {}
        self.available_nodes = {}

    def get_available_modules(self) -> Dict[str, str]:
        """Get list of all available modules"""
        return {
            'loaders': 'Model Loaders (Checkpoint, VAE, LoRA, CLIP)',
            'samplers': 'Samplers (KSampler, KSamplerAdvanced)',
            'conditioning': 'Conditioning (CLIP Text Encode)',
            'latent': 'Latent Space (VAE Encode/Decode, Latent)',
            'image': 'Image Processing (Load/Save Image, Preview)',
            'wan_video': 'WAN Video Generation (TextEncode, Sampler, VAE)',
        }

    def load_module(self, module_name: str) -> bool:
        """
        Dynamically load specified node module

        Args:
            module_name: Module name (e.g. 'wan_video', 'loaders')

        Returns:
            bool: Whether loading succeeded
        """
        if module_name in self.loaded_modules:
            logger.info(f"Module {module_name} already loaded")
            return True

        if module_name not in self.BUILTIN_MODULES:
            logger.error(f"Unknown module: {module_name}")
            return False

        module_path = self.BUILTIN_MODULES[module_name]

        try:
            logger.info(f"Loading module: {module_name} ({module_path})")
            module = importlib.import_module(module_path)

            self.loaded_modules[module_name] = module

            # Extract node classes from module
            self._extract_nodes_from_module(module_name, module)

            logger.info(f"Module {module_name} loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load module {module_name}: {e}")
            return False

    def load_modules(self, module_names: List[str]) -> Dict[str, bool]:
        """
        Load multiple modules in batch

        Args:
            module_names: List of module names

        Returns:
            Dict[str, bool]: Loading result for each module
        """
        results = {}
        for name in module_names:
            results[name] = self.load_module(name)
        return results

    def unload_module(self, module_name: str) -> bool:
        """
        Unload specified module

        Args:
            module_name: Module name

        Returns:
            bool: Whether unloading succeeded
        """
        if module_name not in self.loaded_modules:
            logger.warning(f"Module {module_name} not loaded")
            return False

        try:
            # Remove nodes
            nodes_to_remove = [
                node_name for node_name, info in self.available_nodes.items()
                if info['module'] == module_name
            ]
            for node_name in nodes_to_remove:
                del self.available_nodes[node_name]

            # Remove module
            del self.loaded_modules[module_name]

            logger.info(f"Module {module_name} unloaded")
            return True

        except Exception as e:
            logger.error(f"Failed to unload module {module_name}: {e}")
            return False

    def _extract_nodes_from_module(self, module_name: str, module: Any):
        """Extract node classes from module"""
        # Find all classes inheriting from ComfyUINodeInterface
        for attr_name in dir(module):
            if attr_name.startswith('_'):
                continue

            attr = getattr(module, attr_name)

            # Check if it's a class
            if not isinstance(attr, type):
                continue

            # Check if it has INPUT_TYPES method (ComfyUI node signature)
            if hasattr(attr, 'INPUT_TYPES') and callable(getattr(attr, 'INPUT_TYPES')):
                self.available_nodes[attr_name] = {
                    'class': attr,
                    'module': module_name,
                    'category': getattr(attr, 'CATEGORY', 'unknown'),
                    'description': getattr(attr, 'DESCRIPTION', '')
                }

    def get_node_class(self, node_name: str) -> Optional[type]:
        """Get node class"""
        if node_name in self.available_nodes:
            return self.available_nodes[node_name]['class']
        return None

    def get_loaded_modules(self) -> List[str]:
        """Get list of loaded modules"""
        return list(self.loaded_modules.keys())

    def get_available_nodes(self) -> Dict[str, Dict[str, Any]]:
        """Get all available nodes"""
        return self.available_nodes.copy()

    def get_nodes_by_module(self, module_name: str) -> List[str]:
        """Get all nodes from specified module"""
        return [
            node_name for node_name, info in self.available_nodes.items()
            if info['module'] == module_name
        ]

    def get_module_info(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about all modules"""
        info = {}
        for module_name in self.BUILTIN_MODULES.keys():
            is_loaded = module_name in self.loaded_modules
            nodes = self.get_nodes_by_module(module_name) if is_loaded else []

            info[module_name] = {
                'loaded': is_loaded,
                'node_count': len(nodes),
                'nodes': nodes,
                'description': self.get_available_modules().get(module_name, '')
            }
        return info


# Global node loader instance
node_loader = DynamicNodeLoader()
