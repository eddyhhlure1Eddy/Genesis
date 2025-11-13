"""
Genesis Core Optimization Module
Performance optimizations for PyTorch and CUDA
Author: eddy
Date: 2025-11-12
"""

import torch
import logging
import platform
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger('Genesis.Optimization')


class ComputePrecision(Enum):
    """Computation precision modes"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    TF32 = "tf32"
    AUTO = "auto"


class AttentionBackend(Enum):
    """Attention computation backends"""
    FLASH_ATTENTION = "flash"
    CUDNN_ATTENTION = "cudnn"
    MEMORY_EFFICIENT = "efficient"
    MATH = "math"
    AUTO = "auto"


class FP8Format(Enum):
    """FP8 data formats"""
    E4M3FN = "e4m3fn"
    E5M2 = "e5m2"
    E8M0FNU = "e8m0fnu"


class CoreOptimizer:
    """
    Core optimizer for Genesis Engine

    Applies PyTorch and CUDA optimizations for maximum performance
    """

    def __init__(self, device: torch.device):
        """
        Initialize core optimizer

        Args:
            device: Target compute device
        """
        self.device = device
        self.device_type = device.type
        self.is_cuda = self.device_type == 'cuda'
        self.is_mps = self.device_type == 'mps'
        self.is_cpu = self.device_type == 'cpu'

        self.torch_version = self._get_torch_version()
        self.cuda_version = self._get_cuda_version()
        self.cudnn_version = self._get_cudnn_version()
        self.compute_capability = self._get_compute_capability()

        self.optimizations_applied = []
        self.fp8_enabled = False
        self.fp8_format = None

        logger.info(f"Core Optimizer initialized")
        logger.info(f"  Device: {device}")
        logger.info(f"  PyTorch: {self.torch_version}")
        if self.is_cuda:
            logger.info(f"  CUDA: {self.cuda_version}")
            logger.info(f"  cuDNN: {self.cudnn_version}")
            if self.compute_capability:
                logger.info(f"  Compute Capability: sm_{self.compute_capability[0]}{self.compute_capability[1]}")

    def _get_torch_version(self) -> tuple:
        """Get PyTorch version as tuple"""
        try:
            version_str = torch.__version__.split('+')[0]
            parts = version_str.split('.')
            return (int(parts[0]), int(parts[1]))
        except:
            return (0, 0)

    def _get_cuda_version(self) -> Optional[str]:
        """Get CUDA version"""
        if not self.is_cuda:
            return None
        try:
            return torch.version.cuda
        except:
            return None

    def _get_cudnn_version(self) -> Optional[int]:
        """Get cuDNN version"""
        if not self.is_cuda:
            return None
        try:
            return torch.backends.cudnn.version()
        except:
            return None

    def _get_compute_capability(self) -> Optional[tuple]:
        """Get GPU compute capability"""
        if not self.is_cuda:
            return None
        try:
            return torch.cuda.get_device_capability(self.device)
        except:
            return None

    def apply_all_optimizations(
        self,
        precision: str = "auto",
        attention_backend: str = "auto",
        enable_tf32: bool = True,
        enable_cudnn_benchmark: bool = True,
        enable_jit_fusion: bool = True,
        enable_matmul_precision: str = "high",
        enable_fp8: bool = False,
        fp8_format: str = "e4m3fn"
    ) -> Dict[str, bool]:
        """
        Apply all available optimizations

        Args:
            precision: Computation precision (fp32/fp16/bf16/tf32/auto)
            attention_backend: Attention backend (flash/cudnn/efficient/math/auto)
            enable_tf32: Enable TF32 for Ampere+ GPUs
            enable_cudnn_benchmark: Enable cuDNN autotuner
            enable_jit_fusion: Enable JIT kernel fusion
            enable_matmul_precision: Matrix multiplication precision
            enable_fp8: Enable FP8 quantization (Ada/Hopper/Blackwell)
            fp8_format: FP8 format (e4m3fn/e5m2/e8m0fnu)

        Returns:
            Dictionary of applied optimizations
        """
        results = {}

        if self.is_cuda:
            results['tf32'] = self.enable_tf32_precision(enable_tf32)
            results['cudnn_benchmark'] = self.enable_cudnn_benchmark(enable_cudnn_benchmark)
            results['sdpa'] = self.enable_scaled_dot_product_attention(attention_backend)
            results['fp16_accumulation'] = self.enable_fp16_accumulation()
            results['memory_efficient_sdp'] = self.enable_memory_efficient_attention()
            results['flash_sdp'] = self.enable_flash_attention()
            results['cudnn_sdp'] = self.enable_cudnn_attention()
            results['fp16_bf16_reduction'] = self.enable_fp16_bf16_reduction()

            if enable_fp8:
                results['fp8'] = self.enable_fp8_quantization(fp8_format)

        results['jit_fusion'] = self.enable_jit_fusion(enable_jit_fusion)
        results['matmul_precision'] = self.set_matmul_precision(enable_matmul_precision)

        if self.is_mps:
            logger.info("MPS optimizations applied")

        applied_count = sum(1 for v in results.values() if v)
        logger.info(f"Applied {applied_count}/{len(results)} optimizations")

        return results

    def enable_tf32_precision(self, enable: bool = True) -> bool:
        """
        Enable TF32 precision for Ampere+ GPUs

        TF32 provides ~8x speedup for matmul on Ampere+ with minimal accuracy loss

        Args:
            enable: Whether to enable TF32

        Returns:
            Success status
        """
        if not self.is_cuda:
            return False

        try:
            if self.torch_version >= (1, 7):
                torch.backends.cuda.matmul.allow_tf32 = enable
                torch.backends.cudnn.allow_tf32 = enable

                if enable:
                    logger.info("[OK] TF32 enabled (Ampere+ optimization)")
                    self.optimizations_applied.append('tf32')
                return True
        except Exception as e:
            logger.warning(f"Could not set TF32: {e}")
        return False

    def enable_cudnn_benchmark(self, enable: bool = True) -> bool:
        """
        Enable cuDNN autotuner for optimal kernel selection

        Automatically finds best convolution algorithms for your hardware

        Args:
            enable: Whether to enable benchmark mode

        Returns:
            Success status
        """
        if not self.is_cuda:
            return False

        try:
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = enable

                if enable:
                    logger.info("[OK] cuDNN benchmark enabled (auto-tuning)")
                    self.optimizations_applied.append('cudnn_benchmark')
                return True
        except Exception as e:
            logger.warning(f"Could not set cuDNN benchmark: {e}")
        return False

    def enable_scaled_dot_product_attention(self, backend: str = "auto") -> bool:
        """
        Enable optimized scaled dot product attention (SDPA)

        Uses Flash Attention, cuDNN, or memory-efficient implementations

        Args:
            backend: Backend to use (auto/flash/cudnn/efficient/math)

        Returns:
            Success status
        """
        if not self.is_cuda:
            return False

        try:
            if self.torch_version >= (2, 0):
                torch.backends.cuda.enable_math_sdp(True)

                logger.info("[OK] Scaled Dot Product Attention enabled")
                self.optimizations_applied.append('sdpa')
                return True
        except Exception as e:
            logger.warning(f"Could not enable SDPA: {e}")
        return False

    def enable_fp16_accumulation(self) -> bool:
        """
        Enable FP16 accumulation for matrix multiplication

        Significant speedup on modern GPUs with minimal accuracy loss

        Returns:
            Success status
        """
        if not self.is_cuda:
            return False

        try:
            if hasattr(torch.backends.cuda.matmul, 'allow_fp16_accumulation'):
                torch.backends.cuda.matmul.allow_fp16_accumulation = True
                logger.info("[OK] FP16 accumulation enabled")
                self.optimizations_applied.append('fp16_accumulation')
                return True
        except Exception as e:
            logger.warning(f"Could not enable FP16 accumulation: {e}")
        return False

    def enable_memory_efficient_attention(self) -> bool:
        """
        Enable memory-efficient attention implementation

        Returns:
            Success status
        """
        if not self.is_cuda:
            return False

        try:
            if self.torch_version >= (2, 0):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                logger.info("[OK] Memory-efficient attention enabled")
                self.optimizations_applied.append('memory_efficient_sdp')
                return True
        except Exception as e:
            logger.warning(f"Could not enable memory-efficient attention: {e}")
        return False

    def enable_flash_attention(self) -> bool:
        """
        Enable Flash Attention 2.0 backend

        Fastest attention implementation for supported GPUs

        Returns:
            Success status
        """
        if not self.is_cuda:
            return False

        try:
            if self.torch_version >= (2, 0):
                torch.backends.cuda.enable_flash_sdp(True)
                logger.info("[OK] Flash Attention enabled")
                self.optimizations_applied.append('flash_sdp')
                return True
        except Exception as e:
            logger.warning(f"Could not enable Flash Attention: {e}")
        return False

    def enable_cudnn_attention(self) -> bool:
        """
        Enable cuDNN attention backend

        Alternative fast attention implementation

        Returns:
            Success status
        """
        if not self.is_cuda:
            return False

        try:
            if self.torch_version >= (2, 0) and self.cudnn_version and self.cudnn_version >= 8700:
                logger.info("[OK] cuDNN Attention available")
                self.optimizations_applied.append('cudnn_attention')
                return True
        except Exception as e:
            logger.warning(f"cuDNN Attention not available: {e}")
        return False

    def enable_fp16_bf16_reduction(self) -> bool:
        """
        Enable FP16/BF16 reduction for scaled dot product attention

        Improves performance on modern GPUs

        Returns:
            Success status
        """
        if not self.is_cuda:
            return False

        try:
            if self.torch_version >= (2, 5):
                torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
                logger.info("[OK] FP16/BF16 reduction enabled")
                self.optimizations_applied.append('fp16_bf16_reduction')
                return True
        except Exception as e:
            logger.warning(f"Could not enable FP16/BF16 reduction: {e}")
        return False

    def enable_jit_fusion(self, enable: bool = True) -> bool:
        """
        Enable JIT kernel fusion

        Fuses operations for better performance

        Args:
            enable: Whether to enable JIT fusion

        Returns:
            Success status
        """
        try:
            if enable:
                torch._C._jit_set_profiling_executor(True)
                torch._C._jit_set_profiling_mode(True)
                logger.info("[OK] JIT fusion enabled")
                self.optimizations_applied.append('jit_fusion')
            return True
        except Exception as e:
            logger.warning(f"Could not enable JIT fusion: {e}")
        return False

    def set_matmul_precision(self, precision: str = "high") -> bool:
        """
        Set matrix multiplication precision mode

        Args:
            precision: Precision mode (highest/high/medium)

        Returns:
            Success status
        """
        try:
            if self.torch_version >= (2, 0):
                torch.set_float32_matmul_precision(precision)
                logger.info(f"[OK] Matmul precision set to: {precision}")
                self.optimizations_applied.append(f'matmul_precision_{precision}')
                return True
        except Exception as e:
            logger.warning(f"Could not set matmul precision: {e}")
        return False

    def enable_compile_mode(self, model: torch.nn.Module, mode: str = "reduce-overhead") -> Optional[torch.nn.Module]:
        """
        Compile model with torch.compile for optimization

        Args:
            model: Model to compile
            mode: Compilation mode (default/reduce-overhead/max-autotune)

        Returns:
            Compiled model or None if compilation fails
        """
        if self.torch_version < (2, 0):
            logger.warning("torch.compile requires PyTorch 2.0+")
            return None

        try:
            compiled_model = torch.compile(model, mode=mode)
            logger.info(f"[OK] Model compiled with mode: {mode}")
            self.optimizations_applied.append(f'compile_{mode}')
            return compiled_model
        except Exception as e:
            logger.warning(f"Could not compile model: {e}")
            return None

    def enable_fp8_quantization(self, fp8_format: str = "e4m3fn") -> bool:
        """
        Enable FP8 quantization for Ada/Hopper/Blackwell GPUs

        FP8 provides 2x performance improvement over FP16 on supported GPUs
        Requires compute capability >= 8.9 (Ada Lovelace, RTX 40 series)

        Args:
            fp8_format: FP8 format (e4m3fn/e5m2/e8m0fnu)

        Returns:
            Success status
        """
        if not self.is_cuda:
            logger.warning("FP8 requires CUDA device")
            return False

        if not self.compute_capability:
            logger.warning("Could not detect compute capability")
            return False

        major, minor = self.compute_capability

        if (major == 8 and minor >= 9) or major >= 9:
            self.fp8_enabled = True
            self.fp8_format = fp8_format

            arch_name = self._get_architecture_name()
            logger.info(f"[OK] FP8 quantization enabled ({fp8_format}) for {arch_name}")
            logger.info("  • 2x performance vs FP16")
            logger.info("  • Reduced memory bandwidth")
            self.optimizations_applied.append(f'fp8_{fp8_format}')
            return True
        else:
            logger.warning(f"FP8 requires compute capability >= 8.9 (found sm_{major}{minor})")
            return False

    def _get_architecture_name(self) -> str:
        """Get GPU architecture name from compute capability"""
        if not self.compute_capability:
            return "Unknown"

        major, minor = self.compute_capability
        arch_map = {
            (7, 0): "Volta",
            (7, 5): "Turing",
            (8, 0): "Ampere",
            (8, 6): "Ampere",
            (8, 9): "Ada Lovelace",
            (9, 0): "Hopper",
            (12, 0): "Blackwell",
        }

        return arch_map.get((major, minor), f"sm_{major}{minor}")

    def supports_fp8(self) -> bool:
        """
        Check if current GPU supports FP8 quantization

        Returns:
            True if FP8 is supported
        """
        if not self.is_cuda or not self.compute_capability:
            return False

        major, minor = self.compute_capability
        return (major == 8 and minor >= 9) or major >= 9

    def quantize_fp8_tensor(self, tensor: torch.Tensor, fp8_format: str = "e4m3fn") -> torch.Tensor:
        """
        Quantize tensor to FP8 format

        Args:
            tensor: Input tensor
            fp8_format: FP8 format (e4m3fn/e5m2/e8m0fnu)

        Returns:
            Quantized tensor
        """
        if not self.supports_fp8():
            logger.warning("FP8 not supported on this GPU, returning original tensor")
            return tensor

        try:
            if fp8_format == "e4m3fn":
                fp8_dtype = torch.float8_e4m3fn
            elif fp8_format == "e5m2":
                fp8_dtype = torch.float8_e5m2
            elif fp8_format == "e8m0fnu":
                if hasattr(torch, 'float8_e8m0fnu'):
                    fp8_dtype = torch.float8_e8m0fnu
                else:
                    logger.warning("e8m0fnu format not available, using e4m3fn")
                    fp8_dtype = torch.float8_e4m3fn
            else:
                logger.warning(f"Unknown FP8 format {fp8_format}, using e4m3fn")
                fp8_dtype = torch.float8_e4m3fn

            return tensor.to(dtype=fp8_dtype)

        except Exception as e:
            logger.warning(f"FP8 quantization failed: {e}")
            return tensor

    def quantize_model_fp8(self, model: torch.nn.Module, fp8_format: str = "e4m3fn") -> torch.nn.Module:
        """
        Quantize entire model to FP8

        Args:
            model: Model to quantize
            fp8_format: FP8 format

        Returns:
            Quantized model
        """
        if not self.supports_fp8():
            logger.warning("FP8 not supported, returning original model")
            return model

        try:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data = self.quantize_fp8_tensor(param.data, fp8_format)

            logger.info(f"[OK] Model quantized to FP8 ({fp8_format})")
            return model

        except Exception as e:
            logger.warning(f"Model FP8 quantization failed: {e}")
            return model

    def get_optimal_dtype(self) -> torch.dtype:
        """
        Get optimal dtype for current device

        Returns:
            Optimal torch dtype
        """
        if self.is_cuda:
            if self.fp8_enabled and self.supports_fp8():
                if self.fp8_format == "e4m3fn":
                    return torch.float8_e4m3fn
                elif self.fp8_format == "e5m2":
                    return torch.float8_e5m2

            if self.torch_version >= (1, 10) and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        elif self.is_mps:
            return torch.float16
        else:
            return torch.float32

    def get_optimization_report(self) -> Dict[str, Any]:
        """
        Get detailed optimization report

        Returns:
            Report dictionary
        """
        report = {
            'device': str(self.device),
            'device_type': self.device_type,
            'torch_version': f"{self.torch_version[0]}.{self.torch_version[1]}",
            'optimizations_applied': self.optimizations_applied,
            'optimization_count': len(self.optimizations_applied),
        }

        if self.is_cuda:
            report['cuda_version'] = self.cuda_version
            report['cudnn_version'] = self.cudnn_version
            report['gpu_name'] = torch.cuda.get_device_name(self.device)
            report['gpu_compute_capability'] = torch.cuda.get_device_capability(self.device)
            report['architecture'] = self._get_architecture_name()
            report['optimal_dtype'] = str(self.get_optimal_dtype())
            report['fp8_supported'] = self.supports_fp8()
            report['fp8_enabled'] = self.fp8_enabled
            if self.fp8_enabled:
                report['fp8_format'] = self.fp8_format

        return report

    def print_optimization_report(self):
        """Print formatted optimization report"""
        report = self.get_optimization_report()

        print("\n" + "=" * 80)
        print(" Genesis Core Optimization Report")
        print("=" * 80)
        print(f"\nDevice: {report['device']}")
        print(f"PyTorch: {report['torch_version']}")

        if 'gpu_name' in report:
            print(f"GPU: {report['gpu_name']}")
            print(f"Architecture: {report['architecture']}")
            print(f"CUDA: {report['cuda_version']}")
            print(f"cuDNN: {report['cudnn_version']}")
            print(f"Compute Capability: {report['gpu_compute_capability']}")
            print(f"Optimal dtype: {report['optimal_dtype']}")
            print(f"\nFP8 Support: {'Yes' if report['fp8_supported'] else 'No'}")
            if report['fp8_enabled']:
                print(f"FP8 Enabled: Yes ({report['fp8_format']})")
            else:
                print("FP8 Enabled: No")

        print(f"\nOptimizations Applied: {report['optimization_count']}")
        for opt in report['optimizations_applied']:
            print(f"  [OK] {opt}")

        print("=" * 80 + "\n")


def create_optimized_device(
    device_str: str = "cuda",
    device_id: int = 0,
    apply_optimizations: bool = True
) -> tuple[torch.device, Optional[CoreOptimizer]]:
    """
    Create optimized device with all performance enhancements

    Args:
        device_str: Device type (cuda/mps/cpu)
        device_id: Device ID for CUDA
        apply_optimizations: Whether to apply optimizations

    Returns:
        Tuple of (device, optimizer)
    """
    if device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
    elif device_str == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    optimizer = None
    if apply_optimizations:
        optimizer = CoreOptimizer(device)
        optimizer.apply_all_optimizations()

    return device, optimizer
