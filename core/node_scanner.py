"""
Genesis Node Scanner
Dynamically scan and load nodes from modules (ComfyUI-compatible)
Author: eddy
"""

import os
import sys
import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import traceback

from .node_registry import get_node_registry

logger = logging.getLogger(__name__)


class NodeScanner:
    """
    Node Scanner - Dynamically discover and load nodes

    Features:
    - Scan built-in nodes
    - Scan custom nodes
    - ComfyUI-compatible
    - Error handling and logging
    """

    def __init__(self, registry=None):
        self.registry = registry or get_node_registry()
        self.loaded_modules: Set[str] = set()
        self.failed_modules: Dict[str, str] = {}
        self.logger = logging.getLogger(f"{__name__}.NodeScanner")

    def scan_builtin_nodes(self, builtin_path: str = None) -> int:
        """
        Scan and load built-in nodes

        Args:
            builtin_path: Path to built-in nodes directory

        Returns:
            Number of nodes loaded
        """
        if builtin_path is None:
            # Default to genesis/nodes directory
            genesis_root = Path(__file__).parent.parent
            builtin_path = genesis_root / "nodes"

        builtin_path = Path(builtin_path)
        if not builtin_path.exists():
            self.logger.warning(f"Built-in nodes path not found: {builtin_path}")
            return 0

        self.logger.info(f"Scanning built-in nodes from: {builtin_path}")

        # Get all Python files in nodes directory
        node_files = []
        for file in builtin_path.glob("*.py"):
            if file.name.startswith('_'):
                continue
            node_files.append(file)

        # Load each node module
        count = 0
        for file in node_files:
            module_name = file.stem
            try:
                nodes_loaded = self._load_node_module(
                    file,
                    module_name=f"genesis.nodes.{module_name}",
                    module_label=f"builtin:{module_name}"
                )
                count += nodes_loaded
            except Exception as e:
                self.logger.error(f"Failed to load built-in nodes from {file.name}: {e}")
                self.failed_modules[str(file)] = str(e)

        self.logger.info(f"Loaded {count} built-in nodes from {len(node_files)} modules")
        return count

    def scan_custom_nodes(self, custom_nodes_path: str = None) -> int:
        """
        Scan and load custom nodes

        Args:
            custom_nodes_path: Path to custom nodes directory

        Returns:
            Number of nodes loaded
        """
        if custom_nodes_path is None:
            # Default to genesis/custom_nodes directory
            genesis_root = Path(__file__).parent.parent
            custom_nodes_path = genesis_root / "custom_nodes"

        custom_nodes_path = Path(custom_nodes_path)
        if not custom_nodes_path.exists():
            self.logger.info(f"Custom nodes directory not found: {custom_nodes_path}")
            return 0

        self.logger.info(f"Scanning custom nodes from: {custom_nodes_path}")

        # Scan all subdirectories
        count = 0
        for subdir in custom_nodes_path.iterdir():
            if not subdir.is_dir():
                continue

            # Skip hidden and special directories
            if subdir.name.startswith('.') or subdir.name.startswith('__'):
                continue

            # Check for __init__.py
            init_file = subdir / "__init__.py"
            if not init_file.exists():
                self.logger.debug(f"Skipping {subdir.name}: no __init__.py found")
                continue

            try:
                nodes_loaded = self._load_custom_node_package(subdir)
                count += nodes_loaded
                self.logger.info(f"Loaded {nodes_loaded} nodes from custom node: {subdir.name}")
            except Exception as e:
                self.logger.error(f"Failed to load custom node {subdir.name}: {e}")
                self.logger.debug(traceback.format_exc())
                self.failed_modules[str(subdir)] = str(e)

        self.logger.info(f"Loaded {count} custom nodes total")
        return count

    def _load_node_module(
        self,
        module_path: Path,
        module_name: str,
        module_label: str = None
    ) -> int:
        """
        Load a single node module

        Args:
            module_path: Path to module file
            module_name: Module name for import
            module_label: Label for registry (optional)

        Returns:
            Number of nodes loaded
        """
        if module_label is None:
            module_label = module_name

        self.logger.debug(f"Loading module: {module_name} from {module_path}")

        # Load module
        try:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load module spec from {module_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

        except Exception as e:
            self.logger.error(f"Failed to import module {module_name}: {e}")
            raise

        # Check for NODE_CLASS_MAPPINGS
        if not hasattr(module, 'NODE_CLASS_MAPPINGS'):
            self.logger.debug(f"No NODE_CLASS_MAPPINGS found in {module_name}")
            return 0

        node_mappings = module.NODE_CLASS_MAPPINGS
        if not isinstance(node_mappings, dict):
            self.logger.warning(f"NODE_CLASS_MAPPINGS in {module_name} is not a dict")
            return 0

        # Register all nodes
        count = 0
        for node_name, node_class in node_mappings.items():
            try:
                self.registry.register(node_name, node_class, module=module_label)
                count += 1
            except Exception as e:
                self.logger.error(f"Failed to register node {node_name}: {e}")

        # Check for NODE_DISPLAY_NAME_MAPPINGS
        if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            display_names = module.NODE_DISPLAY_NAME_MAPPINGS
            for node_name, display_name in display_names.items():
                node_info = self.registry.get(node_name)
                if node_info:
                    node_class = node_info.node_class
                    if not hasattr(node_class, 'DISPLAY_NAME'):
                        node_class.DISPLAY_NAME = display_name

        self.loaded_modules.add(module_name)
        return count

    def _load_custom_node_package(self, package_path: Path) -> int:
        """
        Load a custom node package

        Args:
            package_path: Path to package directory

        Returns:
            Number of nodes loaded
        """
        package_name = package_path.name
        module_name = f"genesis_custom_nodes.{package_name}"

        # Add parent directory to path
        parent_dir = str(package_path.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        # Load the package
        init_file = package_path / "__init__.py"
        return self._load_node_module(
            init_file,
            module_name=module_name,
            module_label=f"custom:{package_name}"
        )

    def load_single_file(self, file_path: str, module_name: str = None) -> int:
        """
        Load nodes from a single Python file

        Args:
            file_path: Path to Python file
            module_name: Module name (auto-generated if None)

        Returns:
            Number of nodes loaded
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if module_name is None:
            module_name = f"genesis_external.{file_path.stem}"

        return self._load_node_module(
            file_path,
            module_name=module_name,
            module_label=f"external:{file_path.stem}"
        )

    def discover_all(
        self,
        builtin_path: str = None,
        custom_nodes_path: str = None,
        additional_paths: List[str] = None
    ) -> Dict[str, int]:
        """
        Discover and load all nodes

        Args:
            builtin_path: Path to built-in nodes
            custom_nodes_path: Path to custom nodes
            additional_paths: Additional paths to scan

        Returns:
            Dictionary with loading statistics
        """
        stats = {
            'builtin': 0,
            'custom': 0,
            'additional': 0,
            'total': 0,
            'failed': 0
        }

        # Load built-in nodes
        try:
            stats['builtin'] = self.scan_builtin_nodes(builtin_path)
        except Exception as e:
            self.logger.error(f"Failed to scan built-in nodes: {e}")

        # Load custom nodes
        try:
            stats['custom'] = self.scan_custom_nodes(custom_nodes_path)
        except Exception as e:
            self.logger.error(f"Failed to scan custom nodes: {e}")

        # Load additional paths
        if additional_paths:
            for path in additional_paths:
                try:
                    path_obj = Path(path)
                    if path_obj.is_file():
                        nodes = self.load_single_file(path)
                        stats['additional'] += nodes
                    elif path_obj.is_dir():
                        # Try to load as custom node package
                        nodes = self._load_custom_node_package(path_obj)
                        stats['additional'] += nodes
                except Exception as e:
                    self.logger.error(f"Failed to load from {path}: {e}")

        stats['total'] = stats['builtin'] + stats['custom'] + stats['additional']
        stats['failed'] = len(self.failed_modules)

        return stats

    def get_load_report(self) -> Dict[str, Any]:
        """Get detailed loading report"""
        return {
            'loaded_modules': sorted(list(self.loaded_modules)),
            'failed_modules': self.failed_modules.copy(),
            'total_loaded': len(self.loaded_modules),
            'total_failed': len(self.failed_modules),
            'registry_stats': self.registry.get_statistics()
        }

    def reload_module(self, module_name: str) -> int:
        """
        Reload a specific module

        Args:
            module_name: Module name to reload

        Returns:
            Number of nodes loaded
        """
        if module_name not in sys.modules:
            raise ValueError(f"Module {module_name} not loaded")

        # Unregister existing nodes from this module
        nodes_to_unregister = []
        for node_name in self.registry.list_all():
            node_info = self.registry.get(node_name)
            if node_info and node_info.module and module_name in node_info.module:
                nodes_to_unregister.append(node_name)

        for node_name in nodes_to_unregister:
            self.registry.unregister(node_name)

        # Reload module
        module = sys.modules[module_name]
        importlib.reload(module)

        # Re-register nodes
        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            node_mappings = module.NODE_CLASS_MAPPINGS
            count = 0
            for node_name, node_class in node_mappings.items():
                try:
                    self.registry.register(node_name, node_class, module=module_name)
                    count += 1
                except Exception as e:
                    self.logger.error(f"Failed to register node {node_name}: {e}")

            self.logger.info(f"Reloaded {count} nodes from {module_name}")
            return count

        return 0


# Global scanner instance
_global_scanner = None


def get_node_scanner() -> NodeScanner:
    """Get global node scanner instance"""
    global _global_scanner
    if _global_scanner is None:
        _global_scanner = NodeScanner()
    return _global_scanner


def discover_and_load_all_nodes(**kwargs) -> Dict[str, int]:
    """Convenience function to discover and load all nodes"""
    scanner = get_node_scanner()
    return scanner.discover_all(**kwargs)
