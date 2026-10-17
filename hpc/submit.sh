#!/bin/bash
# On the Wolffe login node, from the repo folder:  bash hpc/submit.sh
# Lists the remaining work in hpc/tasks.txt and submits one array task per line.
# Safe to run again: finished images are skipped, so it only picks up what is left.
set -euo pipefail
cd "$(dirname "$0")/.."

TOTAL=3669       # test images per model
STABILITY=200    # first images, explained under three seeds
CHUNKS=8         # ranges per model and explainer after the stability slice
STEP=$(( (TOTAL - STABILITY + CHUNKS - 1) / CHUNKS ))

# Task list: the stability slice, then the rest of the split in chunks.
# Ranges never overlap, so every task can run at the same time.
: > hpc/tasks.txt
for job in "finetuned resnet101" "finetuned resnet152" "baseline resnet50"; do
  for stage in lime_run shap_run; do
    echo "$stage $job 0 $STABILITY 0 1 2" >> hpc/tasks.txt
    for (( first = STABILITY; first < TOTAL; first += STEP )); do
      echo "$stage $job $first $(( first + STEP < TOTAL ? first + STEP : TOTAL )) 0" >> hpc/tasks.txt
    done
  done
done

# Submit, at most MAX_GPUS tasks running at once
mkdir -p results/logs
TASKS=$(wc -l < hpc/tasks.txt)
sbatch --array=0-$(( TASKS - 1 ))%"${MAX_GPUS:-8}" hpc/explain.sbatch
