"""
Compare outputs from Qwen3-Instruct, Qwen3-Think, and Qwen2.5-Instruct on the same prompt.
"""

import os
import sys

# Add parent directory to Python path so we can import models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from models import Qwen3Instruct, Qwen3Think, Qwen25Instruct

PROMPT = """There are two ducks in front of a duck, 
        two ducks behind a duck and a duck in the middle. 
        How many ducks are there?"""
# Answer: 3

CACHE_DIR = os.getenv('CACHE_DIR', '/home/adewinmb/orcd/scratch')

# Load all three models
print("=" * 80)
print("Loading Qwen3-VL-8B-Instruct...")
print("=" * 80)
model_qwen3_inst = Qwen3Instruct.load_model(
    cache_dir=CACHE_DIR,
    dtype="auto",
    device_map="auto",
    freeze_params=True
)

print("\n" + "=" * 80)
print("Loading Qwen3-VL-8B-Thinking...")
print("=" * 80)
model_qwen3_think = Qwen3Think.load_model(
    cache_dir=CACHE_DIR,
    dtype="auto",
    device_map="auto",
    freeze_params=True
)

print("\n" + "=" * 80)
print("Loading Qwen2.5-VL-7B-Instruct...")
print("=" * 80)
model_qwen25_inst = Qwen25Instruct.load_model(
    cache_dir=CACHE_DIR,
    dtype="auto",
    device_map="auto",
    freeze_params=True
)

# Create a black image
image = Image.new("RGB", (224, 224), color="black")

# Generate outputs from all models
print("\n" + "=" * 80)
print("PROMPT:")
print("=" * 80)
print(PROMPT)

print("\n" + "=" * 80)
print("Qwen3-VL-8B-Instruct Response:")
print("=" * 80)
output_qwen3_inst = model_qwen3_inst.generate(PROMPT, image, max_new_tokens=1000)
print(output_qwen3_inst)

print("\n" + "=" * 80)
print("Qwen3-VL-8B-Thinking Response:")
print("=" * 80)
output_qwen3_think = model_qwen3_think.generate(PROMPT, image, max_new_tokens=1000)
print(output_qwen3_think)

print("\n" + "=" * 80)
print("Qwen2.5-VL-7B-Instruct Response:")
print("=" * 80)
output_qwen25_inst = model_qwen25_inst.generate(PROMPT, image, max_new_tokens=1000)
print(output_qwen25_inst)

print("\n" + "=" * 80)
print("Comparison Complete!")
print("=" * 80)

