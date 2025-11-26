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

# Example of running training script and overriding a parameter. See all parameters in config/default.yaml
python exp/adversarial_image.py --model-name qwen25inst
