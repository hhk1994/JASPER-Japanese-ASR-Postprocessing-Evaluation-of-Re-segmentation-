#!/bin/bash
#SBATCH --job-name=tok_%j
#SBATCH --output=logs/tok_%x_%j.log
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

MODE=$1

echo "Starting tokenization in mode: $MODE"
python3 Japanese_Word_Segmentation.py --mode "$MODE"
echo "Finished tokenization: $MODE"

