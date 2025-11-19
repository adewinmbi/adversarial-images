from typing import Optional
import torch
from torch import Tensor
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor

from .model import AbstractModel


class Llava(AbstractModel):
    """LLaVA model wrapper."""
    
    MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"
    
    def __init__(self):
        super().__init__()
    
    @classmethod
    def load_model(
        cls,
        cache_dir: Optional[str] = None,
        dtype: str = "auto",
        device_map: str = "auto",
        freeze_params: bool = True
    ) -> "Llava":
        """Load LLaVA model and processor."""
        instance = cls()
        
        print(f"Loading {cls.MODEL_NAME}...", flush=True)
        instance.model = LlavaForConditionalGeneration.from_pretrained(
            cls.MODEL_NAME,
            torch_dtype=dtype if dtype != "auto" else torch.float16,
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
        
        Args:
            text: Input text/instruction
            image: PIL Image
            add_generation_prompt: Whether to add generation prompt
            
        Returns:
            Logits tensor of shape [batch, seq_len, vocab_size]
        """
        # LLaVA uses a different prompt format
        if add_generation_prompt:
            prompt = f"USER: <image>\n{text}\nASSISTANT:"
        else:
            prompt = f"USER: <image>\n{text}"
        
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
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
        
        Args:
            text: Input text/instruction
            image: PIL Image
            max_new_tokens: Maximum number of tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text string
        """
        prompt = f"USER: <image>\n{text}\nASSISTANT:"
        
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            # Trim the input tokens
            input_len = inputs['input_ids'].shape[1]
            generated_ids_trimmed = generated_ids[:, input_len:]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
        
        return output_text
