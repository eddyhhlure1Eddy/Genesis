"""
Genesis Triton Kernels
Custom Triton kernels for acceleration
Author: eddy
"""

import torch
import logging

logger = logging.getLogger('Genesis.Triton')

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
    logger.info("Triton available")
except ImportError:
    TRITON_AVAILABLE = False
    logger.warning("Triton not available, using PyTorch fallback")


if TRITON_AVAILABLE:
    
    @triton.jit
    def add_kernel(
        x_ptr, y_ptr, output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr
    ):
        """Triton kernel for element-wise addition"""
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        
        tl.store(output_ptr + offsets, output, mask=mask)
    
    
    @triton.jit
    def softmax_kernel(
        input_ptr, output_ptr,
        input_row_stride,
        output_row_stride,
        n_cols,
        BLOCK_SIZE: tl.constexpr
    ):
        """Triton kernel for softmax"""
        row_idx = tl.program_id(0)
        row_start_ptr = input_ptr + row_idx * input_row_stride
        
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
    
    
    @triton.jit
    def layernorm_kernel(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        input_row_stride, output_row_stride,
        n_cols, eps,
        BLOCK_SIZE: tl.constexpr
    ):
        """Triton kernel for layer normalization"""
        row_idx = tl.program_id(0)
        
        row_start_ptr = input_ptr + row_idx * input_row_stride
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        input_ptrs = row_start_ptr + col_offsets
        row = tl.load(input_ptrs, mask=mask, other=0.0)
        
        mean = tl.sum(row, axis=0) / n_cols
        var = tl.sum((row - mean) * (row - mean), axis=0) / n_cols
        rstd = 1 / tl.sqrt(var + eps)
        
        row_normalized = (row - mean) * rstd
        
        weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
        
        output = row_normalized * weight + bias
        
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, output, mask=mask)


class TritonOps:
    """
    Triton-accelerated operations
    Fallback to PyTorch if Triton not available
    """
    
    @staticmethod
    def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Element-wise addition using Triton
        """
        if not TRITON_AVAILABLE or not x.is_cuda:
            return x + y
        
        try:
            output = torch.empty_like(x)
            n_elements = output.numel()
            
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
            
            return output
        except Exception as e:
            logger.warning(f"Triton add failed: {e}, using PyTorch fallback")
            return x + y
    
    @staticmethod
    def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Softmax using Triton
        """
        if not TRITON_AVAILABLE or not x.is_cuda or dim != -1:
            return torch.softmax(x, dim=dim)
        
        try:
            n_rows, n_cols = x.shape
            output = torch.empty_like(x)
            
            BLOCK_SIZE = triton.next_power_of_2(n_cols)
            num_warps = 4
            if BLOCK_SIZE >= 2048:
                num_warps = 8
            if BLOCK_SIZE >= 4096:
                num_warps = 16
            
            softmax_kernel[(n_rows,)](
                x, output,
                x.stride(0), output.stride(0),
                n_cols,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps
            )
            
            return output
        except Exception as e:
            logger.warning(f"Triton softmax failed: {e}, using PyTorch fallback")
            return torch.softmax(x, dim=dim)
    
    @staticmethod
    def layernorm(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-5
    ) -> torch.Tensor:
        """
        Layer normalization using Triton
        """
        if not TRITON_AVAILABLE or not x.is_cuda:
            return torch.nn.functional.layer_norm(x, x.shape[-1:], weight, bias, eps)
        
        try:
            n_rows, n_cols = x.shape
            output = torch.empty_like(x)
            
            BLOCK_SIZE = triton.next_power_of_2(n_cols)
            
            layernorm_kernel[(n_rows,)](
                x, weight, bias, output,
                x.stride(0), output.stride(0),
                n_cols, eps,
                BLOCK_SIZE=BLOCK_SIZE
            )
            
            return output
        except Exception as e:
            logger.warning(f"Triton layernorm failed: {e}, using PyTorch fallback")
            return torch.nn.functional.layer_norm(x, x.shape[-1:], weight, bias, eps)


def get_triton_config() -> dict:
    """Get Triton configuration and capabilities"""
    if not TRITON_AVAILABLE:
        return {'available': False}
    
    config = {
        'available': True,
        'version': triton.__version__ if hasattr(triton, '__version__') else 'unknown'
    }
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        config['device'] = torch.cuda.get_device_name(0)
        config['compute_capability'] = f"{props.major}.{props.minor}"
        config['max_shared_memory'] = props.max_shared_memory_per_block_optin
    
    return config
