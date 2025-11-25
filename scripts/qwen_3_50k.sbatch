#!/bin/bash         
#SBATCH -N 1                        
#SBATCH -n 4                        
#SBATCH --gres=gpu:h200:1          
#SBATCH -p mit_normal_gpu
#SBATCH -t 6:00:00
#SBATCH --mem=128G

# Set environment variables
export CACHE_DIR='/home/adewinmb/orcd/scratch'

module load miniforge
source activate chimera
uv sync
source .venv/bin/activate

python exp/adversarial_image.py \
    --model-name qwen3inst \
    --num-epochs 500 \
    --log-dir '20251121_instruct3'
