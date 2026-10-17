#!/bin/bash
#SBATCH --job-name=xai-explain
#SBATCH --partition=ampere24
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/%x_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --requeue
# One array task = one line of hpc/tasks.txt: stage model arch first last seeds...
# Submitted by hpc/submit.sh. A requeued or resubmitted task resumes image by image.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Environment
module load Python/Python3.12.3
export OPENSSL_CONF=/dev/null   # skip the cluster OpenSSL config, which breaks the Python ssl module
source .venv/bin/activate
export TORCH_HOME="$PWD/data/torch" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PETS_DEVICE=cuda   # fail loudly instead of falling back to the processor

# This task's range
read -r STAGE MODEL ARCH FIRST LAST SEEDS <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" hpc/tasks.txt)"
echo "$(date '+%F %T') $(hostname): $STAGE $MODEL $ARCH $FIRST-$LAST seeds $SEEDS"

python code/"$STAGE".py --model "$MODEL" --arch "$ARCH" --split test \
  --first "$FIRST" --last "$LAST" --seeds $SEEDS --batch 128
