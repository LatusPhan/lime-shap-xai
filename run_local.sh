#!/usr/bin/env bash
# The whole local run in one command. Nothing is retrained: the checkpoints in
# results/checkpoints are used as they are.
#
#   bash run_local.sh              every model: ResNet-50 and the baseline, then the other depths
#   bash run_local.sh resnet101    one fine-tuned depth
#   mkdir -p results/logs && nohup caffeinate -is bash run_local.sh >> results/logs/run_local.log 2>&1 &
#
# Every stage resumes, so stopping this and starting it again costs at most the images in
# flight. BATCH=128 PARALLEL=4 bash run_local.sh suits a machine with a GPU.
set -uo pipefail
cd "$(dirname "$0")"
TARGET="${1:-all}"
BATCH="${BATCH:-64}"           # images per forward pass; 128 on a GPU
PARALLEL="${PARALLEL:-3}"      # ranges at once; the explainers are limited by the processor
STABILITY="${STABILITY:-200}"  # images explained under three seeds before the rest
PY="${PY:-python}"
DEPTHS=(resnet18 resnet34 resnet101 resnet152)
mkdir -p results/logs

step () { echo "=== $(date '+%a %H:%M') $*"; }
explain () { "$PY" code/run_all.py explain --batch "$BATCH" --parallel "$PARALLEL" --stability "$STABILITY" "$@"; }

"$PY" - <<'PYEOF' || exit 1
import importlib.util, sys
missing = [name for name in ("torch", "torchvision", "lime", "shap", "skimage", "scipy", "pandas", "matplotlib")
           if importlib.util.find_spec(name) is None]
if missing:
    sys.exit("missing: " + ", ".join(missing) + "\ninstall with: pip install -r code/requirements.txt")
import numpy, skimage, torch
if int(numpy.__version__.split(".")[0]) < 2 or not skimage.__version__.startswith("0.25"):
    sys.exit(f"found numpy {numpy.__version__} and scikit-image {skimage.__version__}; "
             "code/requirements.txt needs numpy 2 and scikit-image 0.25.2")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"stack complete: torch {torch.__version__} on {device}")
PYEOF

step "reproducing the 24 stored LIME records"
"$PY" code/lime_run.py --check 2>&1 | tee results/logs/check.log || exit 1

if [ "$TARGET" = all ]; then
  step "ResNet-50 and the stock baseline: LIME and SHAP over the test split"
  explain --models finetuned baseline --archs resnet50 || exit 1
  step "summaries and figures for those two"  # a drawing problem must not stop the depths
  "$PY" code/run_all.py report --models finetuned baseline --archs resnet50 || step "report failed, carrying on"
  step "the other four depths"
  explain --models finetuned --archs "${DEPTHS[@]}" || exit 1
  step "summaries and figures for every model"
  "$PY" code/run_all.py report --models finetuned baseline --archs resnet50 "${DEPTHS[@]}" || exit 1
else
  step "LIME and SHAP over the test split for $TARGET"
  explain --archs "$TARGET" || exit 1
  step "summaries and figures"
  "$PY" code/run_all.py report --archs "$TARGET" || exit 1
fi
step "done: results/summary and results/figures"
