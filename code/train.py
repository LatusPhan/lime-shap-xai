"""Stage 1: fine-tune one ResNet depth on Oxford-IIIT Pet.

The backbone is frozen for the first epoch so a random head cannot wreck the features,
then trained at a tenth of the head's rate under decoupled weight decay.

The best epoch is chosen on a validation slice carved out of trainval, not on the test
split. Checkpoints made before this file were selected on test, which flatters them.
Pass --val-fraction 0 to reproduce that older behaviour.

Never retrain to fix a downstream problem: an explanation only means something against
the weights it came from.

    python train.py --arch resnet50 --epochs 10
    python train.py --arch resnet152 --epochs 15 --workers 0   # workers 0 if the loaders hang
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import runner
from data import NUM_CLASSES, PetDataset, download
from model import ARCHITECTURES, build_model, get_device


def evaluate(model, loader, device):
    """Top-1 accuracy."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += int((model(images).argmax(1) == labels).sum())
            total += labels.numel()
    return correct / max(total, 1)


def train(arch, epochs, batch_size, learning_rate, seed, workers, val_fraction):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()
    print(f"training {arch} on {device.type}")

    # a validation slice from trainval, so the best epoch is not chosen on the test split
    augmented = PetDataset(runner.DATA_ROOT, split="train", augment=True)
    plain = PetDataset(runner.DATA_ROOT, split="train", augment=False)
    order = np.random.default_rng(seed).permutation(len(augmented))
    cut = int(len(order) * val_fraction)
    train_set = Subset(augmented, order[cut:].tolist())
    validation = PetDataset(runner.DATA_ROOT, split="test", augment=False) if cut == 0 \
        else Subset(plain, order[:cut].tolist())
    test_set = PetDataset(runner.DATA_ROOT, split="test", augment=False)

    # decoding and augmenting a batch costs about as much as the forward and backward pass
    options = dict(num_workers=workers, persistent_workers=workers > 0)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **options)
    validation_loader = DataLoader(validation, batch_size=batch_size, **options)
    test_loader = DataLoader(test_set, batch_size=batch_size, **options)
    print(f"{len(train_set)} training, {len(validation)} validation, {len(test_set)} test images, "
          f"{workers} loader workers")

    model = build_model(arch=arch).to(device)
    criterion = nn.CrossEntropyLoss()
    head = list(model.fc.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [{"params": head, "lr": learning_rate},
         {"params": backbone, "lr": learning_rate / 10}], weight_decay=1e-4)

    runner.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    checkpoint = runner.CHECKPOINTS / f"{arch}_pets.pt"
    best = 0.0
    for epoch in range(epochs):
        for parameter in backbone:  # head only for the first epoch
            parameter.requires_grad_(epoch > 0)
        model.train()
        total_loss = seen = 0
        started = time.perf_counter()
        for step, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item() * labels.numel()
            seen += labels.numel()
            if step % 10 == 0 or seen == len(train_set):
                rate = seen / max(time.perf_counter() - started, 1e-6)
                print(f"\r  epoch {epoch + 1}/{epochs}  {seen}/{len(train_set)}  "
                      f"loss {total_loss / seen:.4f}  {rate:.0f} img/s      ", end="", flush=True)
        accuracy = evaluate(model, validation_loader, device)
        print(f"\repoch {epoch + 1}/{epochs}  loss {total_loss / len(train_set):.4f}  "
              f"validation accuracy {accuracy:.4f}  "
              f"({(time.perf_counter() - started) / 60:.1f} min)        ")
        if accuracy >= best:
            best = accuracy
            torch.save({"state_dict": model.state_dict(), "num_classes": NUM_CLASSES,
                        "arch": arch, "val_accuracy": accuracy, "epoch": epoch + 1,
                        "seed": seed, "val_fraction": val_fraction}, checkpoint)
            print(f"  saved {checkpoint.name}")

    print(f"best validation accuracy {best:.4f}")
    # the saved epoch, not the last one, is what every later stage uses
    model.load_state_dict(torch.load(checkpoint, map_location=device,
                                     weights_only=False)["state_dict"])
    print(f"test accuracy of the saved checkpoint {evaluate(model, test_loader, device):.4f}")
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=ARCHITECTURES, default="resnet50")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4, help="loader processes; 0 on Windows if they hang")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="share of trainval held out to pick the best epoch; 0 uses the test split")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    if args.data_root:
        runner.DATA_ROOT = Path(args.data_root)
    download(runner.DATA_ROOT)
    train(args.arch, args.epochs, args.batch_size, args.learning_rate, args.seed,
          args.workers, args.val_fraction)
