from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor
from typing import Optional, List, Dict, Any
from PIL import Image

class AbstractModel(ABC, nn.Module):
    """Abstract base class for all VLMs used in this project."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None
    
    @classmethod
    @abstractmethod
    def load_model(
        cls,
        cache_dir: Optional[str] = None,
        dtype: str = "auto",
        device_map: str = "auto",
        freeze_params: bool = True
    ) -> "AbstractModel":
        """
        Load model and processor.
        
        Args:
            cache_dir: Directory to cache model files
            dtype: Data type for model ("auto", "float16", "bfloat16", etc.)
            device_map: Device mapping strategy ("auto", "cuda", etc.)
            freeze_params: Whether to freeze model parameters
            
        Returns:
            Instance of the model wrapper
        """
        ...
    
    @abstractmethod
    def get_logits(
        self,
        text: str,
        image: Image.Image,
        add_generation_prompt: bool = True
    ) -> Tensor:
        """
        Get logits from PIL input image and text string.
        
        Args:
            text: Input text/instruction
            image: PIL Image
            add_generation_prompt: Whether to add generation prompt
            
        Returns:
            Logits tensor of shape [batch, seq_len, vocab_size]
        """
        ...
    
    @abstractmethod
    def generate(
        self,
        text: str,
        image: Image.Image,
        max_new_tokens: int = 128,
        **kwargs
    ) -> str:
        """
        Generate text output from image and text input.
        
        Args:
            text: Input text/instruction
            image: PIL Image
            max_new_tokens: Maximum number of tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text string
        """
        ...
    
    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        if self.model is not None:
            return next(self.model.parameters()).device
        return torch.device("cpu")
    
    def freeze_parameters(self):
        """Freeze all model parameters."""
        if self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
            # Force CUDA synchronization if available
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    def unfreeze_parameters(self):
        """Unfreeze all model parameters."""
        if self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = True
    