#!/bin/bash         
#SBATCH -N 1                        
#SBATCH -n 4                        
#SBATCH --gres=gpu:h200:1          
#SBATCH -p mit_preemptable
#SBATCH -t 02:00:00
#SBATCH --mem=128G

# Set environment variables
export CACHE_DIR='/home/adewinmb/orcd/scratch'

module load miniforge
source activate chimera
uv sync
source .venv/bin/activate

# Run with default config (qwen3inst)
python exp/adversarial_image.py

# Example: Use LLaVA model
# python exp/adversarial_image.py --model-name llava

# Example: Use Qwen3-Thinking model
# python exp/adversarial_image.py --model-name qwen3think

# Example: Use custom config file
# python exp/adversarial_image.py --config configs/llava.yaml

# Example: Override specific parameters
# python exp/adversarial_image.py --num-epochs 50 --learning-rate 0.001 --model-name llava