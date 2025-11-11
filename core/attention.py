"""
Genesis Advanced Attention Mechanisms
Support for Flash Attention, Sage Attention, SDPA, xFormers
Author: eddy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
import logging
from enum import Enum


class AttentionBackend(Enum):
    """Available attention backends"""
    FLASH_ATTN = "flash_attn"
    SAGE_ATTN = "sage_attn"
    SDPA = "sdpa"
    XFORMERS = "xformers"
    PYTORCH = "pytorch"


class AttentionCapability:
    """Check available attention backends"""
    
    def __init__(self):
        self.flash_attn_available = self._check_flash_attn()
        self.sage_attn_available = self._check_sage_attn()
        self.sdpa_available = self._check_sdpa()
        self.xformers_available = self._check_xformers()
    
    def _check_flash_attn(self) -> bool:
        """Check Flash Attention availability"""
        try:
            from flash_attn import flash_attn_func
            return True
        except ImportError:
            return False
    
    def _check_sage_attn(self) -> bool:
        """Check Sage Attention availability"""
        try:
            import sageattention
            return True
        except ImportError:
            return False
    
    def _check_sdpa(self) -> bool:
        """Check SDPA (Scaled Dot Product Attention) availability"""
        return hasattr(F, 'scaled_dot_product_attention')
    
    def _check_xformers(self) -> bool:
        """Check xFormers availability"""
        try:
            import xformers
            import xformers.ops
            return True
        except ImportError:
            return False
    
    def get_available_backends(self) -> list:
        """Get list of available backends"""
        backends = [AttentionBackend.PYTORCH]
        
        if self.flash_attn_available:
            backends.insert(0, AttentionBackend.FLASH_ATTN)
        
        if self.sage_attn_available:
            backends.insert(0, AttentionBackend.SAGE_ATTN)
        
        if self.sdpa_available:
            backends.append(AttentionBackend.SDPA)
        
        if self.xformers_available:
            backends.append(AttentionBackend.XFORMERS)
        
        return backends
    
    def get_best_backend(self) -> AttentionBackend:
        """Get best available backend (priority order)"""
        if self.flash_attn_available:
            return AttentionBackend.FLASH_ATTN
        elif self.sage_attn_available:
            return AttentionBackend.SAGE_ATTN
        elif self.xformers_available:
            return AttentionBackend.XFORMERS
        elif self.sdpa_available:
            return AttentionBackend.SDPA
        else:
            return AttentionBackend.PYTORCH


class UnifiedAttention:
    """
    Unified attention interface supporting multiple backends
    """
    
    def __init__(self, preferred_backend: Optional[str] = None):
        self.logger = logging.getLogger('Genesis.Attention')
        self.capability = AttentionCapability()
        
        # Select backend
        if preferred_backend:
            self.backend = AttentionBackend(preferred_backend)
            if self.backend not in self.capability.get_available_backends():
                self.logger.warning(f"Backend {preferred_backend} not available, using best available")
                self.backend = self.capability.get_best_backend()
        else:
            self.backend = self.capability.get_best_backend()
        
        self.logger.info(f"Selected attention backend: {self.backend.value}")
        
        # Load backend-specific functions
        self._load_backend()
    
    def _load_backend(self):
        """Load backend-specific functions"""
        if self.backend == AttentionBackend.FLASH_ATTN:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
            self.flash_attn_func = flash_attn_func
            self.flash_attn_varlen_func = flash_attn_varlen_func
        
        elif self.backend == AttentionBackend.SAGE_ATTN:
            import sageattention
            self.sage_attn = sageattention
        
        elif self.backend == AttentionBackend.XFORMERS:
            import xformers.ops as xops
            self.xformers_ops = xops
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        scale: Optional[float] = None,
        is_causal: bool = False
    ) -> torch.Tensor:
        """
        Unified attention forward pass
        
        Args:
            query: Query tensor [B, H, N, D] or [B, N, H, D]
            key: Key tensor [B, H, N, D] or [B, N, H, D]
            value: Value tensor [B, H, N, D] or [B, N, H, D]
            attn_mask: Attention mask
            dropout_p: Dropout probability
            scale: Attention scale
            is_causal: Use causal masking
            
        Returns:
            Attention output
        """
        try:
            if self.backend == AttentionBackend.FLASH_ATTN:
                return self._flash_attn_forward(query, key, value, attn_mask, dropout_p, scale, is_causal)
            
            elif self.backend == AttentionBackend.SAGE_ATTN:
                return self._sage_attn_forward(query, key, value, attn_mask, dropout_p, scale, is_causal)
            
            elif self.backend == AttentionBackend.XFORMERS:
                return self._xformers_forward(query, key, value, attn_mask, dropout_p, scale)
            
            elif self.backend == AttentionBackend.SDPA:
                return self._sdpa_forward(query, key, value, attn_mask, dropout_p, scale, is_causal)
            
            else:
                return self._pytorch_forward(query, key, value, attn_mask, dropout_p, scale, is_causal)
        
        except Exception as e:
            self.logger.warning(f"Attention backend {self.backend.value} failed: {e}, falling back to PyTorch")
            return self._pytorch_forward(query, key, value, attn_mask, dropout_p, scale, is_causal)
    
    def _flash_attn_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        dropout_p: float,
        scale: Optional[float],
        is_causal: bool
    ) -> torch.Tensor:
        """Flash Attention forward"""
        # Flash attention expects [B, N, H, D]
        if query.dim() == 4 and query.shape[1] != query.shape[2]:
            # Assume [B, H, N, D], need to transpose
            q = query.transpose(1, 2).contiguous()
            k = key.transpose(1, 2).contiguous()
            v = value.transpose(1, 2).contiguous()
        else:
            q, k, v = query, key, value
        
        output = self.flash_attn_func(
            q, k, v,
            dropout_p=dropout_p,
            softmax_scale=scale,
            causal=is_causal
        )
        
        # Transpose back if needed
        if query.dim() == 4 and query.shape[1] != query.shape[2]:
            output = output.transpose(1, 2)
        
        return output
    
    def _sage_attn_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        dropout_p: float,
        scale: Optional[float],
        is_causal: bool
    ) -> torch.Tensor:
        """Sage Attention forward"""
        # Sage attention interface (may vary by version)
        if scale is None:
            scale = 1.0 / (query.size(-1) ** 0.5)
        
        output = self.sage_attn.attention(
            query, key, value,
            scale=scale,
            dropout=dropout_p,
            causal=is_causal
        )
        
        return output
    
    def _xformers_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        dropout_p: float,
        scale: Optional[float]
    ) -> torch.Tensor:
        """xFormers memory efficient attention"""
        output = self.xformers_ops.memory_efficient_attention(
            query, key, value,
            attn_bias=attn_mask,
            p=dropout_p,
            scale=scale
        )
        return output
    
    def _sdpa_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        dropout_p: float,
        scale: Optional[float],
        is_causal: bool
    ) -> torch.Tensor:
        """PyTorch SDPA (Scaled Dot Product Attention)"""
        return F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            scale=scale,
            is_causal=is_causal
        )
    
    def _pytorch_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        dropout_p: float,
        scale: Optional[float],
        is_causal: bool
    ) -> torch.Tensor:
        """Standard PyTorch attention"""
        if scale is None:
            scale = 1.0 / (query.size(-1) ** 0.5)
        
        # Compute attention scores
        attn = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        # Apply causal mask if needed
        if is_causal:
            seq_len = query.size(-2)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
                diagonal=1
            )
            attn = attn.masked_fill(causal_mask, float('-inf'))
        
        # Apply attention mask
        if attn_mask is not None:
            attn = attn + attn_mask
        
        # Softmax
        attn = F.softmax(attn, dim=-1)
        
        # Dropout
        if dropout_p > 0:
            attn = F.dropout(attn, p=dropout_p)
        
        # Output
        output = torch.matmul(attn, value)
        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with unified backend support
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        backend: Optional[str] = None
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Unified attention
        self.attention = UnifiedAttention(backend)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            query: Query tensor [B, N, D]
            key: Key tensor [B, N, D]
            value: Value tensor [B, N, D]
            attn_mask: Attention mask
            is_causal: Use causal masking
            
        Returns:
            Output tensor [B, N, D]
        """
        batch_size = query.size(0)
        
        # Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head: [B, N, D] -> [B, H, N, D/H]
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn_output = self.attention.forward(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
        )
        
        # Reshape: [B, H, N, D/H] -> [B, N, D]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.embed_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


def benchmark_attention_backends(
    batch_size: int = 2,
    seq_len: int = 512,
    num_heads: int = 8,
    head_dim: int = 64,
    device: str = 'cuda',
    num_iterations: int = 100
) -> dict:
    """
    Benchmark different attention backends
    
    Returns:
        Dictionary with timing results
    """
    import time
    
    results = {}
    
    # Create test tensors
    embed_dim = num_heads * head_dim
    query = torch.randn(batch_size, seq_len, embed_dim, device=device)
    key = torch.randn(batch_size, seq_len, embed_dim, device=device)
    value = torch.randn(batch_size, seq_len, embed_dim, device=device)
    
    capability = AttentionCapability()
    
    # Test each backend
    for backend in capability.get_available_backends():
        try:
            attn = UnifiedAttention(backend.value)
            
            # Reshape for attention
            q = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            
            # Warmup
            for _ in range(10):
                _ = attn.forward(q, k, v)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(num_iterations):
                _ = attn.forward(q, k, v)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = (time.perf_counter() - start) / num_iterations * 1000
            results[backend.value] = elapsed
            
        except Exception as e:
            results[backend.value] = f"Error: {e}"
    
    return results


def get_attention_info() -> dict:
    """Get attention backend information"""
    capability = AttentionCapability()
    
    return {
        'flash_attn': capability.flash_attn_available,
        'sage_attn': capability.sage_attn_available,
        'sdpa': capability.sdpa_available,
        'xformers': capability.xformers_available,
        'available_backends': [b.value for b in capability.get_available_backends()],
        'best_backend': capability.get_best_backend().value
    }
