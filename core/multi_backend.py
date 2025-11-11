"""
Genesis Multi-Backend Support
Support for GPU, CPU, NPU, CANN, ONNX and multi-device inference
Author: eddy
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Union
import logging
from enum import Enum
import platform


class BackendType(Enum):
    """Supported backend types"""
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"
    NPU = "npu"
    CANN = "cann"
    ONNX = "onnx"
    MULTI_GPU = "multi_gpu"


class DeviceCapability:
    """Device capability information"""
    
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.cuda_device_count = torch.cuda.device_count() if self.cuda_available else 0
        self.mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        self.npu_available = self._check_npu()
        self.cann_available = self._check_cann()
        self.onnx_available = self._check_onnx()
    
    def _check_npu(self) -> bool:
        """Check NPU (Ascend) availability"""
        try:
            import torch_npu
            return torch.npu.is_available()
        except ImportError:
            return False
    
    def _check_cann(self) -> bool:
        """Check CANN availability"""
        try:
            import torch_npu
            return hasattr(torch_npu, 'npu') and torch_npu.npu.is_available()
        except ImportError:
            return False
    
    def _check_onnx(self) -> bool:
        """Check ONNX Runtime availability"""
        try:
            import onnxruntime
            return True
        except ImportError:
            return False
    
    def get_available_backends(self) -> List[BackendType]:
        """Get list of available backends"""
        backends = [BackendType.CPU]
        
        if self.cuda_available:
            backends.append(BackendType.CUDA)
            if self.cuda_device_count > 1:
                backends.append(BackendType.MULTI_GPU)
        
        if self.mps_available:
            backends.append(BackendType.MPS)
        
        if self.npu_available or self.cann_available:
            backends.append(BackendType.NPU)
            backends.append(BackendType.CANN)
        
        if self.onnx_available:
            backends.append(BackendType.ONNX)
        
        return backends


class MultiDeviceManager:
    """
    Advanced multi-device and multi-backend manager
    """
    
    def __init__(self, preferred_backend: Optional[str] = None):
        self.logger = logging.getLogger('Genesis.MultiDevice')
        self.capability = DeviceCapability()
        
        # Select backend
        self.backend = self._select_backend(preferred_backend)
        self.device = self._get_device()
        self.device_ids = []
        
        # Multi-GPU setup
        if self.backend == BackendType.CUDA and self.capability.cuda_device_count > 1:
            self.device_ids = list(range(self.capability.cuda_device_count))
            self.logger.info(f"Multi-GPU enabled: {self.device_ids}")
        
        self.logger.info(f"Selected backend: {self.backend.value}")
        self.logger.info(f"Primary device: {self.device}")
    
    def _select_backend(self, preferred: Optional[str]) -> BackendType:
        """Select best available backend"""
        available = self.capability.get_available_backends()
        
        if preferred:
            preferred_backend = BackendType(preferred.lower())
            if preferred_backend in available:
                return preferred_backend
            else:
                self.logger.warning(f"Preferred backend {preferred} not available")
        
        # Priority order
        if BackendType.CUDA in available:
            return BackendType.CUDA
        elif BackendType.NPU in available:
            return BackendType.NPU
        elif BackendType.MPS in available:
            return BackendType.MPS
        else:
            return BackendType.CPU
    
    def _get_device(self) -> torch.device:
        """Get torch device"""
        if self.backend == BackendType.CUDA:
            return torch.device('cuda:0')
        elif self.backend == BackendType.NPU:
            return torch.device('npu:0')
        elif self.backend == BackendType.MPS:
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information"""
        info = {
            'backend': self.backend.value,
            'device': str(self.device),
            'available_backends': [b.value for b in self.capability.get_available_backends()]
        }
        
        if self.backend == BackendType.CUDA:
            info['cuda_version'] = torch.version.cuda
            info['device_count'] = self.capability.cuda_device_count
            info['devices'] = []
            
            for i in range(self.capability.cuda_device_count):
                props = torch.cuda.get_device_properties(i)
                info['devices'].append({
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'compute_capability': f"{props.major}.{props.minor}",
                    'total_memory_gb': props.total_memory / 1e9
                })
        
        elif self.backend == BackendType.NPU:
            try:
                import torch_npu
                info['npu_count'] = torch.npu.device_count()
            except:
                pass
        
        return info
    
    def distribute_model(self, model: nn.Module) -> nn.Module:
        """
        Distribute model across multiple devices
        
        Args:
            model: PyTorch model
            
        Returns:
            Distributed model
        """
        if self.backend == BackendType.CUDA and len(self.device_ids) > 1:
            # Use DataParallel or DistributedDataParallel
            self.logger.info(f"Distributing model across GPUs: {self.device_ids}")
            model = nn.DataParallel(model, device_ids=self.device_ids)
            return model.to(self.device)
        else:
            return model.to(self.device)
    
    def get_optimal_dtype(self) -> torch.dtype:
        """Get optimal dtype for current backend"""
        if self.backend == BackendType.CUDA:
            # Check if Ampere or newer
            props = torch.cuda.get_device_properties(0)
            if props.major >= 8:
                return torch.bfloat16
            return torch.float16
        
        elif self.backend == BackendType.NPU:
            return torch.float16
        
        elif self.backend == BackendType.CPU:
            return torch.bfloat16
        
        return torch.float32


class NPUAccelerator:
    """
    NPU/CANN acceleration for Huawei Ascend
    """
    
    def __init__(self):
        self.logger = logging.getLogger('Genesis.NPU')
        self.available = self._check_availability()
        
        if self.available:
            self._setup_npu()
    
    def _check_availability(self) -> bool:
        """Check NPU availability"""
        try:
            import torch_npu
            return torch.npu.is_available()
        except ImportError:
            self.logger.warning("torch_npu not available")
            return False
    
    def _setup_npu(self):
        """Setup NPU optimizations"""
        try:
            import torch_npu
            
            # Enable NPU optimizations
            torch_npu.npu.set_compile_mode(jit_compile=True)
            
            self.logger.info("NPU optimizations enabled")
        except Exception as e:
            self.logger.error(f"Failed to setup NPU: {e}")
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Optimize model for NPU"""
        if not self.available:
            return model
        
        try:
            import torch_npu
            model = model.to('npu:0')
            
            # Apply NPU-specific optimizations
            model = torch_npu.npu.amp.initialize(model)
            
            self.logger.info("Model optimized for NPU")
            return model
        except Exception as e:
            self.logger.error(f"NPU optimization failed: {e}")
            return model


class ONNXBackend:
    """
    ONNX Runtime backend for cross-platform inference
    """
    
    def __init__(self, device: str = 'cuda'):
        self.logger = logging.getLogger('Genesis.ONNX')
        self.available = self._check_availability()
        self.device = device
        self.session = None
        
        if self.available:
            self._setup_providers()
    
    def _check_availability(self) -> bool:
        """Check ONNX Runtime availability"""
        try:
            import onnxruntime as ort
            return True
        except ImportError:
            self.logger.warning("ONNX Runtime not available")
            return False
    
    def _setup_providers(self):
        """Setup execution providers"""
        try:
            import onnxruntime as ort
            
            # Get available providers
            available_providers = ort.get_available_providers()
            self.logger.info(f"Available ONNX providers: {available_providers}")
            
            # Select providers based on device
            if self.device == 'cuda' and 'CUDAExecutionProvider' in available_providers:
                self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            elif self.device == 'npu' and 'CANNExecutionProvider' in available_providers:
                self.providers = ['CANNExecutionProvider', 'CPUExecutionProvider']
            else:
                self.providers = ['CPUExecutionProvider']
            
            self.logger.info(f"Using ONNX providers: {self.providers}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup ONNX providers: {e}")
    
    def export_model(self, model: nn.Module, dummy_input: torch.Tensor, 
                    output_path: str, **export_kwargs):
        """
        Export PyTorch model to ONNX
        
        Args:
            model: PyTorch model
            dummy_input: Example input tensor
            output_path: Output ONNX file path
            **export_kwargs: Additional export arguments
        """
        try:
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                **export_kwargs
            )
            
            self.logger.info(f"Model exported to ONNX: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"ONNX export failed: {e}")
            return False
    
    def load_model(self, model_path: str) -> bool:
        """
        Load ONNX model
        
        Args:
            model_path: Path to ONNX model
            
        Returns:
            Success status
        """
        if not self.available:
            return False
        
        try:
            import onnxruntime as ort
            
            self.session = ort.InferenceSession(
                model_path,
                providers=self.providers
            )
            
            self.logger.info(f"ONNX model loaded: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load ONNX model: {e}")
            return False
    
    def infer(self, input_data: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Any:
        """
        Run ONNX inference
        
        Args:
            input_data: Input tensor or dict of tensors
            
        Returns:
            Inference output
        """
        if self.session is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Convert input
            if isinstance(input_data, torch.Tensor):
                input_dict = {'input': input_data.cpu().numpy()}
            else:
                input_dict = {k: v.cpu().numpy() for k, v in input_data.items()}
            
            # Run inference
            outputs = self.session.run(None, input_dict)
            
            return outputs
            
        except Exception as e:
            self.logger.error(f"ONNX inference failed: {e}")
            raise


class MultiGPUManager:
    """
    Multi-GPU inference manager
    """
    
    def __init__(self):
        self.logger = logging.getLogger('Genesis.MultiGPU')
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.devices = [torch.device(f'cuda:{i}') for i in range(self.device_count)]
        
        if self.device_count > 1:
            self.logger.info(f"Multi-GPU mode: {self.device_count} GPUs detected")
    
    def is_available(self) -> bool:
        """Check if multi-GPU is available"""
        return self.device_count > 1
    
    def data_parallel(self, model: nn.Module, device_ids: Optional[List[int]] = None) -> nn.Module:
        """
        Apply DataParallel to model
        
        Args:
            model: PyTorch model
            device_ids: List of GPU IDs to use
            
        Returns:
            DataParallel wrapped model
        """
        if not self.is_available():
            self.logger.warning("Multi-GPU not available")
            return model
        
        if device_ids is None:
            device_ids = list(range(self.device_count))
        
        model = nn.DataParallel(model, device_ids=device_ids)
        self.logger.info(f"DataParallel enabled on GPUs: {device_ids}")
        
        return model
    
    def distributed_setup(self, rank: int, world_size: int, backend: str = 'nccl'):
        """
        Setup distributed training
        
        Args:
            rank: Process rank
            world_size: Total number of processes
            backend: Communication backend
        """
        try:
            import torch.distributed as dist
            
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
            torch.cuda.set_device(rank)
            
            self.logger.info(f"Distributed setup: rank {rank}/{world_size}")
            
        except Exception as e:
            self.logger.error(f"Distributed setup failed: {e}")
    
    def get_memory_info(self) -> List[Dict[str, float]]:
        """Get memory info for all GPUs"""
        info = []
        
        for i in range(self.device_count):
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            free, total = torch.cuda.mem_get_info(i)
            
            info.append({
                'gpu_id': i,
                'allocated_gb': allocated / 1e9,
                'reserved_gb': reserved / 1e9,
                'free_gb': free / 1e9,
                'total_gb': total / 1e9
            })
        
        return info
    
    def balance_load(self, batch_size: int) -> List[int]:
        """
        Calculate batch sizes for load balancing
        
        Args:
            batch_size: Total batch size
            
        Returns:
            List of batch sizes per GPU
        """
        if not self.is_available():
            return [batch_size]
        
        # Simple equal distribution
        base_size = batch_size // self.device_count
        remainder = batch_size % self.device_count
        
        sizes = [base_size] * self.device_count
        for i in range(remainder):
            sizes[i] += 1
        
        return sizes


def get_backend_info() -> Dict[str, Any]:
    """Get comprehensive backend information"""
    capability = DeviceCapability()
    
    info = {
        'cuda': {
            'available': capability.cuda_available,
            'device_count': capability.cuda_device_count
        },
        'mps': {
            'available': capability.mps_available
        },
        'npu': {
            'available': capability.npu_available
        },
        'cann': {
            'available': capability.cann_available
        },
        'onnx': {
            'available': capability.onnx_available
        },
        'available_backends': [b.value for b in capability.get_available_backends()]
    }
    
    # Add CUDA details
    if capability.cuda_available:
        info['cuda']['devices'] = []
        for i in range(capability.cuda_device_count):
            info['cuda']['devices'].append({
                'id': i,
                'name': torch.cuda.get_device_name(i)
            })
    
    return info
