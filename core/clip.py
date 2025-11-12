"""
Genesis CLIP
CLIP text encoder for conditioning
Author: eddy
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union, Any
import logging


class CLIPTextEncoder:
    """
    CLIP text encoder wrapper
    
    Handles text tokenization and encoding for conditioning
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        device: str = "cuda"
    ):
        """
        Initialize CLIP encoder
        
        Args:
            model: CLIP model
            tokenizer: CLIP tokenizer
            device: Computing device
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.logger = logging.getLogger('Genesis.CLIP')
        
        # Max token length (SD 1.5/2.x uses 77)
        self.max_length = 77
        
        if self.model:
            self.model.to(self.device)
            self.model.eval()
    
    def tokenize(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Tokenize text
        
        Args:
            text: Text string or list of strings
            
        Returns:
            Token IDs tensor
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        
        # Handle single string
        if isinstance(text, str):
            text = [text]
        
        # Tokenize
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return tokens['input_ids']
    
    def encode(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text to conditioning
        
        Args:
            text: Text string or list of strings
            
        Returns:
            Conditioning tensor [B, 77, 768] or [B, 77, 1024] for SDXL
        """
        if self.model is None:
            raise RuntimeError("CLIP model not loaded")
        
        # Tokenize
        tokens = self.tokenize(text)
        tokens = tokens.to(self.device)
        
        # Encode
        with torch.no_grad():
            outputs = self.model(tokens)
            
            # Get hidden states
            if hasattr(outputs, 'last_hidden_state'):
                conditioning = outputs.last_hidden_state
            elif hasattr(outputs, 'pooler_output'):
                conditioning = outputs.pooler_output
            else:
                conditioning = outputs
        
        return conditioning
    
    def encode_with_weights(
        self,
        text: str,
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """
        Encode text with token weights (for emphasis)
        
        Args:
            text: Text string
            weights: Weight for each token (1.0 = normal)
            
        Returns:
            Weighted conditioning tensor
        """
        # Basic implementation - can be enhanced with proper weighting
        conditioning = self.encode(text)
        
        if weights is not None:
            weights_tensor = torch.tensor(weights, device=self.device)
            weights_tensor = weights_tensor.unsqueeze(0).unsqueeze(-1)
            conditioning = conditioning * weights_tensor
        
        return conditioning
    
    def encode_prompt_pair(
        self,
        positive_prompt: str,
        negative_prompt: str = ""
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode positive and negative prompts
        
        Args:
            positive_prompt: Positive prompt text
            negative_prompt: Negative prompt text
            
        Returns:
            (positive_conditioning, negative_conditioning)
        """
        positive_cond = self.encode(positive_prompt)
        negative_cond = self.encode(negative_prompt if negative_prompt else "")
        
        return positive_cond, negative_cond
    
    def get_empty_conditioning(self, batch_size: int = 1) -> torch.Tensor:
        """
        Get empty conditioning (for unconditional generation)
        
        Args:
            batch_size: Batch size
            
        Returns:
            Empty conditioning tensor
        """
        return self.encode([""] * batch_size)


class PromptParser:
    """
    Advanced prompt parser
    
    Handles prompt weighting, attention, and special syntax
    """
    
    @staticmethod
    def parse_prompt(prompt: str) -> tuple[List[str], List[float]]:
        """
        Parse prompt with weights
        
        Supports:
        - (word:1.5) - increase weight
        - (word:0.5) - decrease weight
        - [word] - decrease weight (0.9)
        - (word) - increase weight (1.1)
        
        Args:
            prompt: Prompt string with weights
            
        Returns:
            (tokens, weights)
        """
        # Simple implementation - can be enhanced
        tokens = prompt.split()
        weights = [1.0] * len(tokens)
        
        return tokens, weights
    
    @staticmethod
    def parse_attention(prompt: str) -> str:
        """
        Parse and apply attention syntax
        
        Args:
            prompt: Prompt with attention syntax
            
        Returns:
            Processed prompt
        """
        # Remove attention markers for now
        # Can be enhanced to actually apply attention
        import re
        
        # Remove (word:weight) syntax
        prompt = re.sub(r'\(([^:]+):[\d.]+\)', r'\1', prompt)
        
        # Remove [] and () brackets
        prompt = prompt.replace('[', '').replace(']', '')
        prompt = prompt.replace('(', '').replace(')', '')
        
        return prompt
    
    @staticmethod
    def split_prompt_by_length(
        prompt: str,
        max_length: int = 75
    ) -> List[str]:
        """
        Split long prompt into chunks
        
        Args:
            prompt: Long prompt text
            max_length: Maximum tokens per chunk
            
        Returns:
            List of prompt chunks
        """
        words = prompt.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            if len(current_chunk) >= max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
            current_chunk.append(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    @staticmethod
    def combine_prompts(prompts: List[str], separator: str = ", ") -> str:
        """
        Combine multiple prompts
        
        Args:
            prompts: List of prompt strings
            separator: Separator between prompts
            
        Returns:
            Combined prompt
        """
        return separator.join(p.strip() for p in prompts if p.strip())


class PromptEmbedding:
    """Helper for managing prompt embeddings"""
    
    def __init__(self, clip_encoder: CLIPTextEncoder):
        """
        Initialize prompt embedding manager
        
        Args:
            clip_encoder: CLIP encoder instance
        """
        self.clip_encoder = clip_encoder
        self.cache = {}
    
    def get_embedding(
        self,
        prompt: str,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Get prompt embedding with caching
        
        Args:
            prompt: Prompt text
            use_cache: Whether to use cache
            
        Returns:
            Embedding tensor
        """
        if use_cache and prompt in self.cache:
            return self.cache[prompt]
        
        embedding = self.clip_encoder.encode(prompt)
        
        if use_cache:
            self.cache[prompt] = embedding
        
        return embedding
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.cache.clear()
    
    def get_cache_size(self) -> int:
        """Get number of cached embeddings"""
        return len(self.cache)
