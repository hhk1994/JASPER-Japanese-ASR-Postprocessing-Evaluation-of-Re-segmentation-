#!/bin/bash
#SBATCH --job-name=infer_lm_%j
#SBATCH --output=logs/infer_%x_%j.log
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

MODE=$1

echo "Inference using LM for mode: $MODE"
python3 batch_lm_inference.py --single_mode "$MODE"
