"""
Genesis Node Registry System
Dynamic node registration and management system (ComfyUI-compatible)
Author: eddy
"""

import logging
from typing import Dict, Any, List, Optional, Type, Callable
from collections import defaultdict
import inspect

logger = logging.getLogger(__name__)


class NodeInfo:
    """Node information container"""

    def __init__(self, name: str, node_class: Type, module: str = None):
        self.name = name
        self.node_class = node_class
        self.module = module
        self._cached_info = None

    def get_info(self) -> Dict[str, Any]:
        """Get detailed node information"""
        if self._cached_info is not None:
            return self._cached_info

        info = {
            'name': self.name,
            'display_name': self.name,
            'module': self.module,
            'category': 'misc',
            'description': '',
            'input_types': {},
            'return_types': [],
            'return_names': [],
            'function': 'execute',
            'output_node': False,
        }

        # Get INPUT_TYPES
        if hasattr(self.node_class, 'INPUT_TYPES'):
            try:
                input_types_method = getattr(self.node_class, 'INPUT_TYPES')
                if callable(input_types_method):
                    info['input_types'] = input_types_method()
            except Exception as e:
                logger.error(f"Failed to get INPUT_TYPES for {self.name}: {e}")

        # Get RETURN_TYPES
        if hasattr(self.node_class, 'RETURN_TYPES'):
            info['return_types'] = list(self.node_class.RETURN_TYPES)

        # Get RETURN_NAMES
        if hasattr(self.node_class, 'RETURN_NAMES'):
            info['return_names'] = list(self.node_class.RETURN_NAMES)

        # Get FUNCTION
        if hasattr(self.node_class, 'FUNCTION'):
            info['function'] = self.node_class.FUNCTION

        # Get CATEGORY
        if hasattr(self.node_class, 'CATEGORY'):
            info['category'] = self.node_class.CATEGORY

        # Get DESCRIPTION
        if hasattr(self.node_class, 'DESCRIPTION'):
            info['description'] = self.node_class.DESCRIPTION
        elif self.node_class.__doc__:
            info['description'] = self.node_class.__doc__.strip()

        # Get OUTPUT_NODE
        if hasattr(self.node_class, 'OUTPUT_NODE'):
            info['output_node'] = self.node_class.OUTPUT_NODE

        # Get display name
        if hasattr(self.node_class, 'DISPLAY_NAME'):
            info['display_name'] = self.node_class.DISPLAY_NAME

        self._cached_info = info
        return info

    def instantiate(self):
        """Create node instance"""
        return self.node_class()


class NodeRegistry:
    """
    Node Registry - Dynamic node registration and management

    Features:
    - Register nodes dynamically
    - Query nodes by name or category
    - Get node information
    - ComfyUI-compatible
    """

    def __init__(self):
        self._nodes: Dict[str, NodeInfo] = {}
        self._categories: Dict[str, List[str]] = defaultdict(list)
        self._modules: Dict[str, List[str]] = defaultdict(list)
        self._aliases: Dict[str, str] = {}
        self.logger = logging.getLogger(f"{__name__}.NodeRegistry")

    def register(
        self,
        name: str,
        node_class: Type,
        module: str = None,
        aliases: List[str] = None
    ):
        """
        Register a node

        Args:
            name: Node name (unique identifier)
            node_class: Node class
            module: Module name (for organization)
            aliases: Alternative names for the node
        """
        if name in self._nodes:
            self.logger.warning(f"Node '{name}' already registered, overwriting")

        # Create node info
        node_info = NodeInfo(name, node_class, module)
        self._nodes[name] = node_info

        # Register in category index
        info = node_info.get_info()
        category = info.get('category', 'misc')
        if name not in self._categories[category]:
            self._categories[category].append(name)

        # Register in module index
        if module:
            if name not in self._modules[module]:
                self._modules[module].append(name)

        # Register aliases
        if aliases:
            for alias in aliases:
                self._aliases[alias] = name

        self.logger.info(f"Registered node: {name} (category: {category})")

    def register_batch(self, node_mappings: Dict[str, Type], module: str = None):
        """
        Register multiple nodes at once (ComfyUI-compatible)

        Args:
            node_mappings: Dictionary of {node_name: node_class}
            module: Module name
        """
        for name, node_class in node_mappings.items():
            self.register(name, node_class, module)

    def unregister(self, name: str):
        """Unregister a node"""
        if name not in self._nodes:
            self.logger.warning(f"Node '{name}' not found")
            return

        node_info = self._nodes[name]
        info = node_info.get_info()

        # Remove from category index
        category = info.get('category', 'misc')
        if name in self._categories[category]:
            self._categories[category].remove(name)

        # Remove from module index
        if node_info.module and name in self._modules[node_info.module]:
            self._modules[node_info.module].remove(name)

        # Remove from registry
        del self._nodes[name]

        # Remove aliases
        aliases_to_remove = [k for k, v in self._aliases.items() if v == name]
        for alias in aliases_to_remove:
            del self._aliases[alias]

        self.logger.info(f"Unregistered node: {name}")

    def get(self, name: str) -> Optional[NodeInfo]:
        """
        Get node info by name

        Args:
            name: Node name or alias

        Returns:
            NodeInfo object or None
        """
        # Check direct name
        if name in self._nodes:
            return self._nodes[name]

        # Check alias
        if name in self._aliases:
            actual_name = self._aliases[name]
            return self._nodes.get(actual_name)

        return None

    def get_class(self, name: str) -> Optional[Type]:
        """Get node class by name"""
        node_info = self.get(name)
        return node_info.node_class if node_info else None

    def get_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed node information"""
        node_info = self.get(name)
        return node_info.get_info() if node_info else None

    def has(self, name: str) -> bool:
        """Check if node exists"""
        return name in self._nodes or name in self._aliases

    def list_all(self) -> List[str]:
        """Get all registered node names"""
        return sorted(list(self._nodes.keys()))

    def list_by_category(self, category: str = None) -> Dict[str, List[str]]:
        """
        List nodes by category

        Args:
            category: Specific category to list, or None for all

        Returns:
            Dictionary of {category: [node_names]}
        """
        if category:
            return {category: sorted(self._categories.get(category, []))}

        return {
            cat: sorted(nodes)
            for cat, nodes in self._categories.items()
        }

    def list_by_module(self, module: str = None) -> Dict[str, List[str]]:
        """List nodes by module"""
        if module:
            return {module: sorted(self._modules.get(module, []))}

        return {
            mod: sorted(nodes)
            for mod, nodes in self._modules.items()
        }

    def get_categories(self) -> List[str]:
        """Get all categories"""
        return sorted(list(self._categories.keys()))

    def get_all_node_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information for all nodes"""
        return {
            name: node_info.get_info()
            for name, node_info in self._nodes.items()
        }

    def search(self, query: str, search_in: List[str] = None) -> List[str]:
        """
        Search nodes by query

        Args:
            query: Search query
            search_in: Fields to search in ['name', 'category', 'description']

        Returns:
            List of matching node names
        """
        if search_in is None:
            search_in = ['name', 'category', 'description']

        query_lower = query.lower()
        results = []

        for name, node_info in self._nodes.items():
            info = node_info.get_info()

            # Search in name
            if 'name' in search_in and query_lower in name.lower():
                results.append(name)
                continue

            # Search in category
            if 'category' in search_in and query_lower in info.get('category', '').lower():
                results.append(name)
                continue

            # Search in description
            if 'description' in search_in and query_lower in info.get('description', '').lower():
                results.append(name)
                continue

        return sorted(results)

    def clear(self):
        """Clear all registered nodes"""
        self._nodes.clear()
        self._categories.clear()
        self._modules.clear()
        self._aliases.clear()
        self.logger.info("Registry cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            'total_nodes': len(self._nodes),
            'total_categories': len(self._categories),
            'total_modules': len(self._modules),
            'total_aliases': len(self._aliases),
            'nodes_by_category': {
                cat: len(nodes) for cat, nodes in self._categories.items()
            }
        }

    def __len__(self) -> int:
        """Get number of registered nodes"""
        return len(self._nodes)

    def __contains__(self, name: str) -> bool:
        """Check if node exists (support 'in' operator)"""
        return self.has(name)

    def __repr__(self) -> str:
        return f"<NodeRegistry(nodes={len(self._nodes)}, categories={len(self._categories)})>"


# Global node registry instance
_global_registry = None


def get_node_registry() -> NodeRegistry:
    """Get global node registry instance"""
    global _global_registry
    if _global_registry is None:
        _global_registry = NodeRegistry()
    return _global_registry


def register_node(name: str, node_class: Type, **kwargs):
    """Convenience function to register a node"""
    registry = get_node_registry()
    registry.register(name, node_class, **kwargs)


def register_nodes(node_mappings: Dict[str, Type], module: str = None):
    """Convenience function to register multiple nodes"""
    registry = get_node_registry()
    registry.register_batch(node_mappings, module)
