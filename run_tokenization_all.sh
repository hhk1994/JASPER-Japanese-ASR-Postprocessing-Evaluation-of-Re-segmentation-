#!/bin/bash

MODES=("char" "sudachi_A" "sudachi_B" "sudachi_C")

mkdir -p logs

for MODE in "${MODES[@]}"; do
  echo "Submitting tokenization job for mode: $MODE"
  sbatch slurm_scripts/tokenize_job.sh "$MODE"
done

