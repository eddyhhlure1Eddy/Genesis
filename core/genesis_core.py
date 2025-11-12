"""
Genesis Core Engine
Pure execution engine - UI agnostic, completely open
Author: eddy
Date: 2025-11-12
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import torch

logger = logging.getLogger(__name__)


class GenesisCore:
    """
    Genesis Core Engine - Pure execution engine

    Features:
    - UI agnostic (no UI dependencies)
    - Task-based execution (not node-based)
    - Plugin system
    - Multiple execution modes
    - Resource management
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Genesis Core

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.device = self._setup_device()
        self.plugins = {}
        self.models = {}
        self.callbacks = {}
        self.optimizer = None

        self._apply_core_optimizations()

        logger.info(f"Genesis Core initialized (device={self.device})")

    def _setup_device(self) -> torch.device:
        """Setup compute device"""
        device_str = self.config.get('device', 'cuda')

        if device_str == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Using GPU: {gpu_name}")
        elif device_str == 'mps' and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Using Apple Metal (MPS)")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")

        return device

    def _apply_core_optimizations(self):
        """Apply core PyTorch and CUDA optimizations"""
        try:
            from .optimization import CoreOptimizer

            enable_opt = self.config.get('enable_optimizations', True)
            if not enable_opt:
                logger.info("Core optimizations disabled by config")
                return

            self.optimizer = CoreOptimizer(self.device)
            results = self.optimizer.apply_all_optimizations(
                enable_tf32=self.config.get('enable_tf32', True),
                enable_cudnn_benchmark=self.config.get('enable_cudnn_benchmark', True),
                enable_jit_fusion=self.config.get('enable_jit_fusion', True),
                enable_fp8=self.config.get('enable_fp8', False),
                fp8_format=self.config.get('fp8_format', 'e4m3fn')
            )

            applied = sum(1 for v in results.values() if v)
            logger.info(f"Applied {applied}/{len(results)} core optimizations")

            if self.optimizer.fp8_enabled:
                logger.info(f"FP8 quantization enabled: {self.optimizer.fp8_format}")

        except ImportError:
            logger.warning("optimization module not found, skipping optimizations")
        except Exception as e:
            logger.warning(f"Could not apply optimizations: {e}")

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task

        Args:
            task: Task description
                {
                    'type': 'text_to_image',
                    'params': {...},
                    'id': 'optional-task-id'
                }

        Returns:
            Result dictionary
        """
        task_type = task.get('type')
        task_id = task.get('id', 'unknown')

        logger.info(f"Executing task: {task_type} (id={task_id})")

        try:
            # Call callbacks
            self._call_callbacks('before_execute', task)

            # Execute based on type
            if task_type == 'text_to_image':
                result = self._execute_text_to_image(task)
            elif task_type == 'image_to_image':
                result = self._execute_image_to_image(task)
            elif task_type == 'workflow':
                result = self._execute_workflow(task)
            else:
                raise ValueError(f"Unknown task type: {task_type}")

            # Call callbacks
            self._call_callbacks('after_execute', task, result)

            logger.info(f"Task completed: {task_id}")
            return {
                'success': True,
                'task_id': task_id,
                'result': result
            }

        except Exception as e:
            logger.error(f"Task failed: {task_id} - {e}")
            self._call_callbacks('on_error', task, e)

            return {
                'success': False,
                'task_id': task_id,
                'error': str(e)
            }

    def execute_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a complex workflow

        Args:
            workflow: Workflow description
                {
                    'tasks': [
                        {'id': '1', 'type': 'load_model', ...},
                        {'id': '2', 'type': 'encode', 'depends': ['1'], ...},
                        ...
                    ]
                }

        Returns:
            Results dictionary
        """
        return self.execute({'type': 'workflow', 'workflow': workflow})

    def _execute_text_to_image(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute text-to-image task"""
        params = task.get('params', {})
        prompt = params.get('prompt', '')
        steps = params.get('steps', 20)
        size = params.get('size', [512, 512])

        logger.info(f"Generating image: '{prompt[:50]}...' ({steps} steps)")

        # Placeholder implementation
        return {
            'image': None,  # PIL Image or tensor
            'prompt': prompt,
            'steps': steps,
            'size': size,
            'metadata': {
                'seed': params.get('seed', -1),
                'cfg_scale': params.get('cfg_scale', 7.0)
            }
        }

    def _execute_image_to_image(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute image-to-image task"""
        params = task.get('params', {})

        return {
            'image': None,
            'metadata': {}
        }

    def _execute_workflow(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute complex workflow"""
        workflow = task.get('workflow', {})
        tasks = workflow.get('tasks', [])

        # Execute tasks in dependency order
        results = {}
        for subtask in tasks:
            task_id = subtask.get('id')
            depends = subtask.get('depends', [])

            # Wait for dependencies
            for dep in depends:
                if dep not in results:
                    raise ValueError(f"Dependency {dep} not satisfied for task {task_id}")

            # Execute subtask
            result = self.execute(subtask)
            results[task_id] = result

        return {
            'tasks_completed': len(results),
            'results': results
        }

    def register_plugin(self, name: str, plugin: Any):
        """Register a plugin"""
        self.plugins[name] = plugin
        logger.info(f"Plugin registered: {name}")

    def load_model(self, model_name: str, model_type: str = 'checkpoint') -> Any:
        """Load a model"""
        logger.info(f"Loading model: {model_name} (type={model_type})")

        # Placeholder implementation
        model = {
            'name': model_name,
            'type': model_type,
            'loaded': True
        }

        self.models[model_name] = model
        return model

    def unload_model(self, model_name: str):
        """Unload a model"""
        if model_name in self.models:
            del self.models[model_name]
            logger.info(f"Model unloaded: {model_name}")

            # Clear CUDA cache if using GPU
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

    def register_callback(self, event: str, callback: Callable):
        """
        Register event callback

        Events:
        - before_execute
        - after_execute
        - on_error
        - on_progress
        """
        if event not in self.callbacks:
            self.callbacks[event] = []

        self.callbacks[event].append(callback)
        logger.info(f"Callback registered: {event}")

    def _call_callbacks(self, event: str, *args, **kwargs):
        """Call registered callbacks"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Callback error ({event}): {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get core status"""
        return {
            'device': str(self.device),
            'plugins_loaded': len(self.plugins),
            'models_loaded': len(self.models),
            'cuda_available': torch.cuda.is_available(),
            'memory_allocated': self._get_memory_info()
        }

    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information"""
        if self.device.type == 'cuda':
            return {
                'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                'total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)
            }
        return {}

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Genesis Core...")

        # Unload all models
        for model_name in list(self.models.keys()):
            self.unload_model(model_name)

        # Clear CUDA cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        logger.info("Cleanup complete")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()

    def __repr__(self) -> str:
        return f"<GenesisCore(device={self.device}, plugins={len(self.plugins)}, models={len(self.models)})>"
