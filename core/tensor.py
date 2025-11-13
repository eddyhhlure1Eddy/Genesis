"""
Tensor wrapper for Genesis Core

Author: eddy
Date: 2025-11-12
"""

import torch
import numpy as np
from typing import Union, Optional, Any


class TensorWrapper:
    """
    Wrapper class for tensors to provide unified interface
    """

    def __init__(self, data: Union[torch.Tensor, np.ndarray, Any]):
        """
        Initialize tensor wrapper

        Args:
            data: Tensor data (torch.Tensor, numpy array, or other)
        """
        if isinstance(data, torch.Tensor):
            self.tensor = data
        elif isinstance(data, np.ndarray):
            self.tensor = torch.from_numpy(data)
        else:
            # Try to convert to tensor
            self.tensor = torch.tensor(data)

    @property
    def shape(self):
        """Get tensor shape"""
        return tuple(self.tensor.shape)

    @property
    def dtype(self):
        """Get tensor dtype"""
        return self.tensor.dtype

    @property
    def device(self):
        """Get tensor device"""
        return self.tensor.device

    def to(self, device: Union[str, torch.device]) -> 'TensorWrapper':
        """
        Move tensor to device

        Args:
            device: Target device

        Returns:
            New TensorWrapper on target device
        """
        return TensorWrapper(self.tensor.to(device))

    def cpu(self) -> 'TensorWrapper':
        """Move tensor to CPU"""
        return TensorWrapper(self.tensor.cpu())

    def cuda(self) -> 'TensorWrapper':
        """Move tensor to CUDA"""
        return TensorWrapper(self.tensor.cuda())

    def numpy(self) -> np.ndarray:
        """Convert to numpy array"""
        return self.tensor.cpu().numpy()

    def float(self) -> 'TensorWrapper':
        """Convert to float32"""
        return TensorWrapper(self.tensor.float())

    def half(self) -> 'TensorWrapper':
        """Convert to float16"""
        return TensorWrapper(self.tensor.half())

    def __repr__(self) -> str:
        return f"TensorWrapper(shape={self.shape}, dtype={self.dtype}, device={self.device})"

    def __getitem__(self, key):
        """Support indexing"""
        return TensorWrapper(self.tensor[key])

    def __setitem__(self, key, value):
        """Support item assignment"""
        if isinstance(value, TensorWrapper):
            self.tensor[key] = value.tensor
        else:
            self.tensor[key] = value