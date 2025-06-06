#!/bin/bash
#SBATCH --job-name=nemo_asr_%j
#SBATCH --output=logs/asr_%x_%j.log
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6

INPUT_DIR=$1

echo "🎙️ Starting ASR on directory: $INPUT_DIR"
python3 Stage1_NeMo_transcription.py --input_dir "$INPUT_DIR"
echo "Finished ASR on: $INPUT_DIR"
