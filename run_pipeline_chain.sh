#!/bin/bash

set -e

MODES=("char" "sudachi_A" "sudachi_B" "sudachi_C")
TRAIN_JOBS=()
INFER_JOBS=()

mkdir -p logs

echo "Submitting training jobs..."

# 1. Submit training jobs
for MODE in "${MODES[@]}"; do
  JOBID=$(sbatch --parsable slurm_scripts/train_job.sh "$MODE")
  echo "Training job for $MODE submitted with Job ID: $JOBID"
  TRAIN_JOBS+=($JOBID)
done

echo "Submitting inference jobs with dependencies..."

# 2. Submit inference jobs dependent on training jobs
for i in "${!MODES[@]}"; do
  MODE=${MODES[$i]}
  TRAIN_ID=${TRAIN_JOBS[$i]}
  INF_ID=$(sbatch --parsable --dependency=afterok:$TRAIN_ID slurm_scripts/infer_job.sh "$MODE")
  echo "Inference job for $MODE chained to training job $TRAIN_ID as $INF_ID"
  INFER_JOBS+=($INF_ID)
done

# 3. Combine all inference job IDs
ALL_INFER_IDS=$(IFS=:; echo "${INFER_JOBS[*]}")

echo "Submitting WER evaluation after all inference jobs complete..."

# 4. Submit final WER job
sbatch --dependency=afterok:$ALL_INFER_IDS slurm_scripts/wer_job.sh
