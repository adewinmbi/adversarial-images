from typing import Optional
import torch
from torch import Tensor
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from .model import AbstractModel


class Qwen25Instruct32B(AbstractModel):
    """Qwen2.5-VL-32B-Instruct model wrapper."""
    
    MODEL_NAME = "Qwen/Qwen2.5-VL-32B-Instruct"
    
    def __init__(self):
        super().__init__()
    
    @classmethod
    def load_model(
        cls,
        cache_dir: Optional[str] = None,
        dtype: str = "auto",
        device_map: str = "auto",
        freeze_params: bool = True
    ) -> "Qwen25Instruct32B":
        """Load Qwen2.5-VL-32B-Instruct model and processor."""
        instance = cls()
        
        print(f"Loading {cls.MODEL_NAME}...", flush=True)
        instance.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cls.MODEL_NAME,
            torch_dtype=dtype,
            device_map=device_map,
            cache_dir=cache_dir
        )
        instance.processor = AutoProcessor.from_pretrained(
            cls.MODEL_NAME,
            cache_dir=cache_dir
        )
        
        if freeze_params:
            print("Freezing model parameters...", flush=True)
            instance.freeze_parameters()
            print("Model loaded and frozen.", flush=True)
        else:
            print("Model loaded.", flush=True)
        
        return instance
    
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
        image: Image.Image = None,
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
        
        if image is None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                    ],
                }
            ]
        else:
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

