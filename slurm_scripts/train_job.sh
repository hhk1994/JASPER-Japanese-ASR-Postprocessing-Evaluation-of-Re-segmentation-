#!/bin/bash
#SBATCH --job-name=train_lm_%j
#SBATCH --output=logs/train_%x_%j.log
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

MODE=$1
DATA_DIR="/mnt/data"

echo "Training LSTM LM for mode: $MODE"
python3 train_lstm_lm.py \
  --mode "$MODE" \
  --data "$DATA_DIR/train_${MODE}.txt" \
  --vocab "$DATA_DIR/vocab_${MODE}.txt" \
  --checkpoint "$DATA_DIR/lstm_lm_${MODE}.pth"
