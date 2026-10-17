#!/bin/bash
# One-off, on the Wolffe login node, from the repo folder:  bash hpc/setup.sh
# Compute nodes may have no network, so everything is downloaded here first.
set -euo pipefail
cd "$(dirname "$0")/.."

# Python environment
module load Python/Python3.12.3
export OPENSSL_CONF=/dev/null   # skip the cluster OpenSSL config, which breaks the Python ssl module
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126
# lime only ships as source; building it in place avoids the isolated pip that fails SSL on Wolffe
pip install --no-cache-dir setuptools wheel
pip install --no-cache-dir --no-build-isolation -r code/requirements.txt

# Dataset and the stock ImageNet weights the baseline uses
export TORCH_HOME="$PWD/data/torch"
python code/run_all.py download
python -c "from torchvision.models import resnet50, ResNet50_Weights; resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)"
python -c "import torch, lime, shap, skimage; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"
