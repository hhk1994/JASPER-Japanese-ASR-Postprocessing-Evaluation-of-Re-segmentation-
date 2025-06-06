#!/bin/bash

DOMAINS=(
  "/mnt/data/navigation"
  "/mnt/data/messaging"
  "/mnt/data/music_media"
  "/mnt/data/General_Knowledge"
  "/mnt/data/japanese_common_voice_test"
  "/mnt/data/japanese_common_voice_train"
  "/mnt/data/japanese_common_voice_dev"
)

mkdir -p logs

for DIR in "${DOMAINS[@]}"; do
  echo "Submitting ASR job for: $DIR"
  sbatch slurm_scripts/stage1_nemo_asr_one.sh "$DIR"
done

