#!/bin/bash
#SBATCH --job-name=wer_eval
#SBATCH --output=logs/wer_eval_%j.log
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --mem=4G

echo "📊 Running WER domain-separated evaluation..."
python3 wer_eval_domains.py
