# Test the instruction and thinking models on a single prompt.
# Prints both outputs to console.

import os
import sys

# Add parent directory to Python path so we can import models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from models import Qwen3Instruct, Qwen3Think

PROMPT="""There are two ducks in front of a duck, 
        two ducks behind a duck and a duck in the middle. 
        How many ducks are there?"""
# Answer: 3

model_inst = Qwen3Instruct.load_model(
    cache_dir=os.getenv('CACHE_DIR', '/home/adewinmb/orcd/scratch'),
    dtype="auto",
    device_map="auto",
    freeze_params=True
)

model_think = Qwen3Think.load_model(
    cache_dir=os.getenv('CACHE_DIR', '/home/adewinmb/orcd/scratch'),
    dtype="auto",
    device_map="auto",
    freeze_params=True
)

image = Image.new("RGB", (224, 224), color="black")

output_inst = model_inst.generate(
    PROMPT,
    image,
    max_new_tokens=1000
)
print(f"\nInstruction response:\n{output_inst}")

output_think = model_think.generate(
    PROMPT,
    image,
    max_new_tokens=1000
)
print(f"\nThinking response:\n{output_think}")
