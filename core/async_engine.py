"""
Genesis Async Engine
Asynchronous model loading and inference
Author: eddy
"""

import torch
import torch.nn as nn
import asyncio
import threading
from typing import Optional, Dict, Any, Callable, Awaitable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from queue import Queue, Empty
from dataclasses import dataclass
import time


@dataclass
class AsyncTask:
    """Async task representation"""
    task_id: str
    task_type: str
    params: Dict[str, Any]
    callback: Optional[Callable] = None
    priority: int = 0
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class AsyncModelLoader:
    """
    Asynchronous model loading
    Load models in background without blocking
    """
    
    def __init__(self, num_workers: int = 2):
        self.logger = logging.getLogger('Genesis.AsyncLoader')
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.loading_tasks = {}
        self.loaded_models = {}
        
        self.logger.info(f"AsyncModelLoader initialized with {num_workers} workers")
    
    async def load_model_async(
        self,
        model_path: str,
        model_class: type,
        device: str = 'cuda',
        **kwargs
    ) -> str:
        """
        Load model asynchronously
        
        Args:
            model_path: Path to model
            model_class: Model class
            device: Target device
            **kwargs: Additional arguments
            
        Returns:
            Task ID
        """
        import uuid
        task_id = str(uuid.uuid4())
        
        def load_fn():
            try:
                self.logger.info(f"Loading model: {model_path}")
                
                # Load model
                model = model_class(**kwargs)
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
                model = model.to(device)
                model.eval()
                
                self.loaded_models[task_id] = {
                    'model': model,
                    'path': model_path,
                    'device': device,
                    'loaded_at': time.time()
                }
                
                self.logger.info(f"Model loaded: {task_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                return False
        
        # Submit to executor
        future = self.executor.submit(load_fn)
        self.loading_tasks[task_id] = future
        
        return task_id
    
    def get_model(self, task_id: str) -> Optional[nn.Module]:
        """Get loaded model by task ID"""
        if task_id in self.loaded_models:
            return self.loaded_models[task_id]['model']
        return None
    
    def is_loaded(self, task_id: str) -> bool:
        """Check if model is loaded"""
        return task_id in self.loaded_models
    
    def is_loading(self, task_id: str) -> bool:
        """Check if model is still loading"""
        if task_id in self.loading_tasks:
            return not self.loading_tasks[task_id].done()
        return False
    
    async def wait_for_model(self, task_id: str, timeout: float = 60.0) -> Optional[nn.Module]:
        """
        Wait for model to finish loading
        
        Args:
            task_id: Task ID
            timeout: Timeout in seconds
            
        Returns:
            Loaded model or None
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_loaded(task_id):
                return self.get_model(task_id)
            
            if task_id in self.loading_tasks:
                if self.loading_tasks[task_id].done():
                    # Check if successful
                    if self.loading_tasks[task_id].result():
                        return self.get_model(task_id)
                    else:
                        return None
            
            await asyncio.sleep(0.1)
        
        self.logger.warning(f"Model loading timeout: {task_id}")
        return None
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        self.loaded_models.clear()
        self.loading_tasks.clear()


class AsyncInferenceEngine:
    """
    Asynchronous inference engine
    Non-blocking model inference with queue
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda', max_queue_size: int = 100):
        self.logger = logging.getLogger('Genesis.AsyncInference')
        self.model = model
        self.device = torch.device(device)
        self.max_queue_size = max_queue_size
        
        # Task queue
        self.task_queue = Queue(maxsize=max_queue_size)
        self.results = {}
        
        # Worker thread
        self.worker_running = False
        self.worker_thread = None
        
        self.logger.info("AsyncInferenceEngine initialized")
    
    def start(self):
        """Start inference worker"""
        if self.worker_running:
            self.logger.warning("Worker already running")
            return
        
        self.worker_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        self.logger.info("Inference worker started")
    
    def stop(self):
        """Stop inference worker"""
        self.worker_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        
        self.logger.info("Inference worker stopped")
    
    def _worker_loop(self):
        """Worker loop for processing inference tasks"""
        while self.worker_running:
            try:
                # Get task from queue
                task = self.task_queue.get(timeout=1.0)
                
                # Process task
                result = self._process_task(task)
                
                # Store result
                self.results[task.task_id] = result
                
                # Call callback if provided
                if task.callback:
                    task.callback(result)
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
    
    def _process_task(self, task: AsyncTask) -> Dict[str, Any]:
        """Process single inference task"""
        try:
            start_time = time.time()
            
            # Get input
            input_data = task.params.get('input')
            if input_data is None:
                raise ValueError("No input provided")
            
            # Convert to tensor
            if not isinstance(input_data, torch.Tensor):
                input_data = torch.tensor(input_data)
            
            input_data = input_data.to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(input_data)
            
            # Convert to CPU/numpy
            if isinstance(output, torch.Tensor):
                output = output.cpu().numpy()
            
            elapsed = time.time() - start_time
            
            return {
                'task_id': task.task_id,
                'status': 'success',
                'output': output,
                'elapsed_time': elapsed
            }
            
        except Exception as e:
            self.logger.error(f"Task processing failed: {e}")
            return {
                'task_id': task.task_id,
                'status': 'failed',
                'error': str(e)
            }
    
    async def submit_async(
        self,
        input_data: Any,
        callback: Optional[Callable] = None,
        priority: int = 0
    ) -> str:
        """
        Submit inference task asynchronously
        
        Args:
            input_data: Input data
            callback: Callback function
            priority: Task priority (higher = more important)
            
        Returns:
            Task ID
        """
        import uuid
        task_id = str(uuid.uuid4())
        
        task = AsyncTask(
            task_id=task_id,
            task_type='inference',
            params={'input': input_data},
            callback=callback,
            priority=priority
        )
        
        # Add to queue
        try:
            self.task_queue.put(task, block=False)
            return task_id
        except:
            raise RuntimeError("Task queue full")
    
    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get result by task ID"""
        return self.results.get(task_id)
    
    def is_complete(self, task_id: str) -> bool:
        """Check if task is complete"""
        return task_id in self.results
    
    async def wait_for_result(self, task_id: str, timeout: float = 60.0) -> Optional[Dict[str, Any]]:
        """
        Wait for task result
        
        Args:
            task_id: Task ID
            timeout: Timeout in seconds
            
        Returns:
            Task result or None
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_complete(task_id):
                return self.get_result(task_id)
            
            await asyncio.sleep(0.1)
        
        return None


class AsyncBatchProcessor:
    """
    Async batch processor for efficient inference
    Accumulates requests and processes in batches
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        max_batch_size: int = 8,
        max_wait_time: float = 0.1
    ):
        self.logger = logging.getLogger('Genesis.AsyncBatch')
        self.model = model
        self.device = torch.device(device)
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        
        # Batch queue
        self.batch_queue = []
        self.results = {}
        self.queue_lock = threading.Lock()
        
        # Worker
        self.worker_running = False
        self.worker_thread = None
        
        self.logger.info(f"AsyncBatchProcessor initialized (batch_size={max_batch_size})")
    
    def start(self):
        """Start batch processor"""
        if self.worker_running:
            return
        
        self.worker_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        self.logger.info("Batch processor started")
    
    def stop(self):
        """Stop batch processor"""
        self.worker_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
    
    def _worker_loop(self):
        """Worker loop for batch processing"""
        while self.worker_running:
            time.sleep(self.max_wait_time)
            
            # Get batch
            with self.queue_lock:
                if not self.batch_queue:
                    continue
                
                batch = self.batch_queue[:self.max_batch_size]
                self.batch_queue = self.batch_queue[self.max_batch_size:]
            
            # Process batch
            self._process_batch(batch)
    
    def _process_batch(self, batch: list):
        """Process batch of tasks"""
        try:
            # Collect inputs
            inputs = []
            task_ids = []
            
            for task in batch:
                inputs.append(task.params['input'])
                task_ids.append(task.task_id)
            
            # Stack inputs
            batch_input = torch.stack([
                torch.tensor(inp) if not isinstance(inp, torch.Tensor) else inp
                for inp in inputs
            ]).to(self.device)
            
            # Batch inference
            with torch.no_grad():
                batch_output = self.model(batch_input)
            
            # Split results
            outputs = torch.unbind(batch_output, dim=0)
            
            # Store results
            for task_id, output in zip(task_ids, outputs):
                self.results[task_id] = {
                    'task_id': task_id,
                    'status': 'success',
                    'output': output.cpu().numpy()
                }
                
                # Call callback
                task = next((t for t in batch if t.task_id == task_id), None)
                if task and task.callback:
                    task.callback(self.results[task_id])
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            
            # Mark all as failed
            for task in batch:
                self.results[task.task_id] = {
                    'task_id': task.task_id,
                    'status': 'failed',
                    'error': str(e)
                }
    
    async def submit_async(
        self,
        input_data: Any,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Submit task for batch processing
        
        Args:
            input_data: Input data
            callback: Callback function
            
        Returns:
            Task ID
        """
        import uuid
        task_id = str(uuid.uuid4())
        
        task = AsyncTask(
            task_id=task_id,
            task_type='inference',
            params={'input': input_data},
            callback=callback
        )
        
        with self.queue_lock:
            self.batch_queue.append(task)
        
        return task_id
    
    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get result by task ID"""
        return self.results.get(task_id)


async def async_model_warmup(model: nn.Module, input_shape: tuple, device: str = 'cuda', num_iterations: int = 10):
    """
    Async model warmup
    Run model several times to compile kernels
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device
        num_iterations: Number of warmup iterations
    """
    logger = logging.getLogger('Genesis.Warmup')
    logger.info(f"Starting model warmup ({num_iterations} iterations)")
    
    dummy_input = torch.randn(input_shape, device=device)
    
    with torch.no_grad():
        for i in range(num_iterations):
            _ = model(dummy_input)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            await asyncio.sleep(0)  # Yield control
    
    logger.info("Model warmup complete")
