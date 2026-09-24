"""The network under study: build it, load a checkpoint, and answer with probabilities.

`batch_predict` is the only thing every explainer sees. It takes plain uint8 images
and returns class probabilities, so LIME and SHAP know nothing about the model.
Getting the preprocessing wrong here is the classic silent failure, so the shape and
dtype are checked rather than coerced.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (ResNet50_Weights, get_model, get_model_weights,
                                resnet50)

from data import NUM_CLASSES, preprocess_transform

ARCHITECTURES = ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")


def get_device() -> torch.device:
    """An NVIDIA GPU, then Apple Silicon's Metal backend, then the processor.

    Set PETS_DEVICE=cpu to force the processor if a GPU kernel misbehaves.
    """
    forced = os.environ.get("PETS_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True,
                arch: str = "resnet50") -> nn.Module:
    """An ImageNet-pretrained ResNet of the given depth, with a fresh head."""
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {arch!r}; expected one of {ARCHITECTURES}")
    weights = get_model_weights(arch).DEFAULT if pretrained else None
    model = get_model(arch, weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(checkpoint: str | Path, device: torch.device) -> nn.Module:
    """Load a fine-tuned checkpoint. The architecture is read from the file itself."""
    path = Path(checkpoint)
    if not path.is_file():
        raise SystemExit(f"no checkpoint at {path}. Train one first: python train.py --arch <depth>")
    payload = torch.load(path, map_location=device, weights_only=False)
    arch = payload.get("arch", "resnet50")
    model = build_model(payload["num_classes"], pretrained=False, arch=arch)
    model.load_state_dict(payload["state_dict"])
    print(f"loaded {path.name} [{arch}] "
          f"(val accuracy {payload.get('val_accuracy', float('nan')):.4f})")
    return model.to(device).eval()


def make_batch_predict(model: nn.Module, device: torch.device, batch_size: int = 32):
    """The plain function every explainer calls: images in, probabilities out."""
    preprocess = preprocess_transform()
    model = model.to(device).eval()

    def batch_predict(images) -> np.ndarray:
        images = np.asarray(images)
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(
                f"expected a batch shaped (N, H, W, 3), got {images.shape}. A single image "
                "or a channels-first batch would be silently mangled by the transform.")
        if images.dtype != np.uint8:
            raise ValueError(f"expected uint8 images, got {images.dtype}. ToTensor only "
                             "divides by 255 for uint8, so floats would skip normalisation.")

        probabilities = []
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                chunk = images[start:start + batch_size]
                batch = torch.stack([preprocess(image) for image in chunk]).to(device)
                probabilities.append(F.softmax(model(batch), dim=1).cpu().numpy())
        return np.concatenate(probabilities, axis=0)

    return batch_predict


def predictor(model: str, arch: str, checkpoints: Path, device: torch.device, batch: int):
    """batch_predict for the model being studied. Must be the weights that made the prediction.

    baseline is the stock ImageNet ResNet-50, which answers in 1,000 ImageNet classes
    and has never seen a pet breed. finetuned is one of the trained checkpoints.
    """
    if model == "baseline":
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    else:
        net = load_model(Path(checkpoints) / f"{arch}_pets.pt", device)
    return make_batch_predict(net, device, batch)
