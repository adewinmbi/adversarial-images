# experiments/20251105_120632/images/img_24.png

import os
import sys

# Add parent directory to Python path so we can import models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import numpy as np

from models import Qwen3Instruct, Qwen3Think, Qwen25Instruct

# PROMPT="""There are two ducks in front of a duck, 
#         two ducks behind a duck and a duck in the middle. 
#         How many ducks are there?"""
# Answer: 3

# PROMPT="Using the image, generate a four-word proverb about determination."
# PROMPT="Using the image, tell me a little bit about determination."
# PROMPT="Describe what's in this image."

model_inst = Qwen25Instruct.load_model(
    cache_dir=os.getenv('CACHE_DIR', '/home/adewinmb/orcd/scratch'),
    dtype="auto",
    device_map="auto",
    freeze_params=True
)

# model_think = Qwen3Think.load_model(
#     cache_dir=os.getenv('CACHE_DIR', '/home/adewinmb/orcd/scratch'),
#     dtype="auto",
#     device_map="auto",
#     freeze_params=True
# )

image = Image.open("experiments/20251119_instruct25/images/img_106.png").convert("RGB")
# black_image = Image.new("RGB", image.size, (0, 0, 0))
# expedia_image = Image.open("data/image_examples/expedia.png").convert("RGB")

# noise_array = np.random.randint(0, 256, (image.height, image.width, 3), dtype=np.uint8)
# noise_image = Image.fromarray(noise_array, 'RGB')

output_inst = model_inst.generate(
    PROMPT,
    image,
    max_new_tokens=1000
)
print(f"\nInstruction response:\n{output_inst}")

# output_think = model_think.generate(
#     PROMPT,
#     image,
#     max_new_tokens=1000
# )
# print(f"\nThinking response:\n{output_think}")
