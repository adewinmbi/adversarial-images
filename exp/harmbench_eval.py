import os
import sys
import json

# Add parent directory to Python path so we can import models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import numpy as np

from models import Qwen3Instruct, Qwen3Think, Qwen25Instruct, Qwen25Instruct32B

IMG_PATH="experiments/20251202_jailbreak_32b/images/img_500.png"

model = Qwen25Instruct32B.load_model(
    cache_dir=os.getenv('CACHE_DIR', '/home/adewinmb/orcd/scratch'),
    dtype="auto",
    device_map="auto",
    freeze_params=True
)

image = Image.open(IMG_PATH).convert("RGB")

with open('data/harmbench_mini_custom.json', 'r') as f:
    data = json.load(f)

outputs = {"questions": []}

for prompt in data['questions']:
    output = model.generate(
        prompt['prompt'],
        image,
        max_new_tokens=1000
    )
    
    outputs['questions'].append({
        'question_id': prompt['question_id'],
        'prompt': prompt['prompt'],
        'target_category': prompt['target_category'],
        'output': output
    })
    print('Output: ', output[:100])
    
with open('data/harmbench_mini_results.json', 'w') as f:
    json.dump(outputs, f, indent=4)