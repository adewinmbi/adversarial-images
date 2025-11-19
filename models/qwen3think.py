from typing import Optional
import torch
from torch import Tensor
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from .model import AbstractModel


class Qwen3Think(AbstractModel):
    """Qwen3-VL reasoning/thinking model wrapper."""
    
    MODEL_NAME = "Qwen/Qwen3-VL-8B-Thinking"
    
    def __init__(self):
        super().__init__()
    
    @classmethod
    def load_model(
        cls,
        cache_dir: Optional[str] = None,
        dtype: str = "auto",
        device_map: str = "auto",
        freeze_params: bool = True
    ) -> "Qwen3Think":
        """Load Qwen3-VL reasoning model and processor."""
        instance = cls()
        
        print(f"Loading {cls.MODEL_NAME} (reasoning mode)...", flush=True)
        instance.model = Qwen3VLForConditionalGeneration.from_pretrained(
            cls.MODEL_NAME,
            dtype=dtype,
            device_map=device_map,
            cache_dir=cache_dir
        )
        instance.processor = AutoProcessor.from_pretrained(
            cls.MODEL_NAME,
            cache_dir=cache_dir
        )
        
        if freeze_params:
            instance.freeze_parameters()
        
        return instance
    
    def get_logits(
        self,
        text: str,
        image: Image.Image,
        add_generation_prompt: bool = True
    ) -> Tensor:
        """
        Get logits from PIL input image and text string.
        Uses reasoning-oriented prompting strategy.
        
        Args:
            text: Input text/instruction
            image: PIL Image
            add_generation_prompt: Whether to add generation prompt
            
        Returns:
            Logits tensor of shape [batch, seq_len, vocab_size]
        """
        # For reasoning models, we might want to add thinking prompts
        # This can be customized based on the specific reasoning model requirements
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=False)
        
        return outputs.logits
    
    def generate(
        self,
        text: str,
        image: Image.Image,
        max_new_tokens: int = 128,
        **kwargs
    ) -> str:
        """
        Generate text output from image and text input.
        May include reasoning chain/thought process depending on model.
        
        Args:
            text: Input text/instruction
            image: PIL Image
            max_new_tokens: Maximum number of tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text string (may include reasoning steps)
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
        
        return output_text