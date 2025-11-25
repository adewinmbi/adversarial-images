#!/bin/bash         
#SBATCH -N 1                        
#SBATCH -n 4                        
#SBATCH --gres=gpu:h200:1          
#SBATCH -p mit_preemptable
#SBATCH -t 2-00:00:00
#SBATCH --mem=128G

# Set environment variables
export CACHE_DIR='/home/adewinmb/orcd/scratch'

module load miniforge
source activate chimera
uv sync
source .venv/bin/activate

python exp/adversarial_image.py \
    --model-name qwen25inst \
    --num-epochs 500 \
    --dataset-type jailbreak \
    --log-dir experiments/20251121_jailbreak3
