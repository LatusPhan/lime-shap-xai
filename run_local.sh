#!/usr/bin/env bash
# The whole local run in one command. Nothing is retrained: the checkpoints in
# results/checkpoints are used as they are.
#
#   bash run_local.sh resnet50
#   nohup caffeinate -is bash run_local.sh resnet50 > results/logs/run_local.log 2>&1 &
#
# Every stage resumes, so stopping this and starting it again costs at most the image in
# flight. BATCH=128 PARALLEL=4 bash run_local.sh resnet50 suits a machine with a GPU.
set -u
cd "$(dirname "$0")"
ARCH="${1:-resnet50}"
BATCH="${BATCH:-64}"       # images per forward pass; 128 on a GPU
PARALLEL="${PARALLEL:-3}"  # ranges at once; the explainers are limited by the processor
PY="${PY:-python}"
mkdir -p results/logs

step () { echo "=== $(date '+%H:%M') $*"; }

"$PY" - <<'PYEOF' || exit 1
import importlib.util, sys
missing = [name for name in ("torch", "torchvision", "lime", "shap", "skimage", "scipy", "pandas")
           if importlib.util.find_spec(name) is None]
if missing:
    sys.exit("missing: " + ", ".join(missing) + "\ninstall with: pip install -r code/requirements.txt")
print("stack complete")
PYEOF

step "reproducing the 24 stored LIME records"
"$PY" code/lime_run.py --check 2>&1 | tee "results/logs/check_$ARCH.log"

if [ ! -f "results/scores/finetuned_$ARCH.csv" ]; then
  step "scoring $ARCH over the test split"
  "$PY" code/score.py --arch "$ARCH" --batch "$BATCH" 2>&1 | tee "results/logs/score_$ARCH.log"
fi

step "LIME and SHAP over the test split, $PARALLEL ranges at a time"
"$PY" code/run_all.py explain --archs "$ARCH" --batch "$BATCH" --parallel "$PARALLEL" --stability 200

step "summaries and figures"
"$PY" code/report.py --archs "$ARCH" --models finetuned
"$PY" code/figures_overlap.py --arch "$ARCH"
"$PY" code/figures_accuracy.py --arch "$ARCH"
step "done: results/summary and results/figures"
