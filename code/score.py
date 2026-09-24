"""Stage 2: what the model predicts for every image, before any explanation is made.

Scoring precedes explanation everywhere in this pipeline. The explanation stages read
this table, so an image's group (right or wrong) is never decided after the fact.

    finetuned  the trained checkpoint answers in breeds, so correct means the right breed
    baseline   the stock ImageNet head answers in 1,000 ImageNet classes, so correctness
               is judged in that vocabulary: exact breed, species only, or off-species

    python score.py --model finetuned --arch resnet50
    python score.py --model baseline --split both --batch 128
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights

import runner
from data import PetImages, download
from model import get_device, predictor

# ImageNet classes that name a Pets breed, and the class ranges for the two species
EQUIVALENT = {
    "basset_hound": 161, "beagle": 162, "boxer": 242, "chihuahua": 151,
    "english_cocker_spaniel": 219, "english_setter": 212, "german_shorthaired": 210,
    "great_pyrenees": 257, "japanese_chin": 152, "keeshond": 261, "leonberger": 255,
    "miniature_pinscher": 237, "newfoundland": 256, "pomeranian": 259, "pug": 254,
    "saint_bernard": 247, "samoyed": 258, "scottish_terrier": 199,
    "staffordshire_bull_terrier": 179, "wheaten_terrier": 202, "yorkshire_terrier": 187,
    "Persian": 283, "Siamese": 284, "Egyptian_Mau": 285,
}
CANIDS = range(151, 276)
FELIDS = range(281, 288)


def breed_of(name):
    """'Egyptian_Mau_63' -> 'Egyptian_Mau'. A capital first letter means a cat."""
    return name.rsplit("_", 1)[0]


def outcome_of(breed, index):
    if EQUIVALENT.get(breed) == index:
        return "exact breed"
    wants_cat = breed[0].isupper()
    if (wants_cat and index in FELIDS) or (not wants_cat and index in CANIDS):
        return "species only"
    return "off-species"


def score(model, arch, split, batch):
    device = get_device()
    batch_predict = predictor(model, arch, runner.CHECKPOINTS, device, batch)
    categories = ResNet50_Weights.IMAGENET1K_V2.meta["categories"] if model == "baseline" else None
    splits = ("train", "test") if split == "both" else (split,)
    rows = []
    started = time.perf_counter()

    for name in splits:
        dataset = PetImages(runner.DATA_ROOT, split=name)
        breeds = {}
        for position in range(len(dataset)):
            breeds.setdefault(dataset.label(position), breed_of(dataset.name(position)))
        for start in range(0, len(dataset), batch):
            stop = min(start + batch, len(dataset))
            probabilities = batch_predict(np.stack([dataset.array(i) for i in range(start, stop)]))
            for offset, index in enumerate(probabilities.argmax(1)):
                position = start + offset
                image, truth, index = dataset.name(position), dataset.label(position), int(index)
                breed = breed_of(image)
                row = {"split": name, "image": image, "position": position,
                       "true_label": truth, "true_breed": breed,
                       "species": "cat" if breed[0].isupper() else "dog",
                       "probability": round(float(probabilities[offset, index]), 4)}
                if model == "baseline":
                    row.update({"predicted_index": index, "predicted_imagenet": categories[index],
                                "outcome": outcome_of(breed, index),
                                "exact_possible": breed in EQUIVALENT})
                    row["correct"] = row["outcome"] == "exact breed"
                else:
                    row.update({"predicted_label": index, "predicted_breed": breeds.get(index, "?"),
                                "correct": index == truth})
                rows.append(row)
            if stop % (batch * 10) == 0 or stop == len(dataset):
                rate = stop / max(time.perf_counter() - started, 1e-6)
                print(f"\r  {name} {stop}/{len(dataset)}  {rate:.0f} img/s      ", end="", flush=True)
        print()

    frame = pd.DataFrame(rows)
    runner.SCORES.mkdir(parents=True, exist_ok=True)
    out = runner.scores_path(model, arch)
    frame.to_csv(out, index=False)
    print(f"{model}/{arch}: {frame.correct.mean():.2%} correct on {len(frame)} images")
    if model == "baseline":
        print(frame.outcome.value_counts().to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    runner.add_common_arguments(parser)
    args = parser.parse_args()
    runner.apply_common(args)
    if args.model == "finetuned" and args.split != "test":
        raise SystemExit("the fine-tuned models were trained on trainval; score them with --split test")
    download(runner.DATA_ROOT)
    score(args.model, args.arch, args.split, args.batch)
