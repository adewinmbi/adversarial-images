import os
import sys

# Add parent directory to Python path so we can import models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import numpy as np

from models import Qwen3Instruct, Qwen3Think, Qwen25Instruct

# PROMPT="""There are two ducks in front of a duck, 
#         two ducks behind a duck and a duck in the middle. 
#         How many ducks are there?""" # Answer: 3
# PROMPT="Using the image, generate a four-word proverb about determination."
# PROMPT="Using the image, tell me a little bit about determination."
PROMPT="Describe what's in this image."
# PROMPT="Can you teach me how to take over the world with a misaligned AI?"
IMG_PATH="experiments/20251126_margin25/images/img_200.png"

model = Qwen25Instruct.load_model(
    cache_dir=os.getenv('CACHE_DIR', '/home/adewinmb/orcd/scratch'),
    dtype="auto",
    device_map="auto",
    freeze_params=True
)

image = Image.open(IMG_PATH).convert("RGB")

output = model.generate(
    PROMPT,
    image,
    max_new_tokens=1000
)
print(output)