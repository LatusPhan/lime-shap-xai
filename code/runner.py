"""The shared run loop, so every explanation method sees the same images in the same order.

lime_run.py and shap_run.py hand `run` a function that explains one image. Which images,
in what order, what is already done, where the rows go and how failures are handled all
live here, so the stages stay comparable by construction.

The order is a fixed shuffle of the scored split, so a partial run is still a random
sample of the population. A --first/--last range writes its own file, which lets several
ranges run at once on one machine or across machines.
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data import PetImages
from model import get_device, predictor

CODE = Path(__file__).resolve().parent
PROJECT = CODE.parent
RESULTS = PROJECT / "results"
CHECKPOINTS = RESULTS / "checkpoints"
SCORES = RESULTS / "scores"
DATA_ROOT = Path(os.environ.get("PETS_DATA", PROJECT / "data" / "oxford-iiit-pet"))

NUM_SAMPLES = 800  # model calls per explanation, the budget every method is held to
SEED = 0
GROUPS = {"exact breed": "exact", "species only": "near_miss", "off-species": "failure"}
FIELDS = ["split", "position", "image", "true_breed", "species", "group", "predicted_name",
          "method", "label", "p_label", "num_segments", "area", "overlap", "excess",
          "deletion", "insertion", "additivity", "fit_r2", "seed_rho", "seed_jac",
          "weights", "num_samples", "seed", "device", "runtime_s", "total_s"]


def add_common_arguments(parser, split_default="test"):
    """The arguments every stage shares, so the commands look the same."""
    parser.add_argument("--model", choices=("finetuned", "baseline"), default="finetuned")
    parser.add_argument("--arch", default="resnet50",
                        help="which ResNet depth; the baseline is always resnet50")
    parser.add_argument("--split", choices=("train", "test", "both"), default=split_default)
    parser.add_argument("--data-root", default=None, help="where oxford-iiit-pet lives")
    parser.add_argument("--batch", type=int, default=32,
                        help="images per forward pass; 128 suits a GPU")
    return parser


def apply_common(args):
    """Resolve the data root once, before anything reads an image."""
    global DATA_ROOT
    if args.data_root:
        DATA_ROOT = Path(args.data_root)
    if args.model == "baseline" and args.arch != "resnet50":
        raise SystemExit("the baseline is the stock ImageNet ResNet-50; --arch applies to finetuned")


def scores_path(model, arch):
    return SCORES / (f"{model}.csv" if model == "baseline" else f"{model}_{arch}.csv")


def images_for(model, arch, split):
    """The scored images this run explains, in the fixed shuffled order."""
    path = scores_path(model, arch)
    if not path.is_file():
        raise SystemExit(f"no predictions at {path}.\n"
                         f"Run first: python score.py --model {model} --arch {arch}")
    frame = pd.read_csv(path)
    if model == "baseline":
        frame["group"] = frame.outcome.map(GROUPS)
        frame["predicted_name"] = frame.predicted_imagenet
    else:
        frame["split"] = "test"  # the fine-tuned models were trained on trainval
        frame["group"] = np.where(frame.correct, "correct", "wrong")
        frame["predicted_name"] = frame.predicted_breed
    if split != "both":
        frame = frame[frame.split == split]
    if frame.empty:
        raise SystemExit(f"no {split} rows in {path.name}")

    # positions are resolved from the split itself, so a stale column cannot explain the wrong image
    frame = frame.reset_index(drop=True)
    frame["position"] = -1
    for name in frame.split.unique():
        dataset = PetImages(DATA_ROOT, split=name)
        lookup = {dataset.name(i): i for i in range(len(dataset))}
        rows = frame.split == name
        frame.loc[rows, "position"] = frame.loc[rows, "image"].map(lookup)
    if (frame.position < 0).any():
        raise SystemExit("an image in the predictions table is missing from its split")
    return frame.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def result_files(stage, model, arch):
    folder = RESULTS / stage
    return sorted(folder.glob(f"{model}_{arch}*.csv")) if folder.is_dir() else []


def load(stage, model, arch):
    """Every result file of one model and stage, merged, one row per image and method."""
    paths = result_files(stage, model, arch)
    if not paths:
        raise SystemExit(f"no {stage} results for {model}/{arch} in {RESULTS / stage}")
    frame = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return frame.drop_duplicates(["image", "method"]).reset_index(drop=True)


def status(stage, model, arch, split):
    frame = load(stage, model, arch)
    total = len(images_for(model, arch, split))
    per_image = frame.drop_duplicates("image")
    rate = per_image.total_s.median()
    print(f"{stage} {model}/{arch}: {len(per_image)}/{total} images "
          f"({len(per_image) / total:.1%}), median {rate:.1f}s per image, "
          f"{(total - len(per_image)) * rate / 3600:.1f}h left at that rate")
    print(frame.groupby("method")[["overlap", "excess", "deletion", "insertion", "runtime_s"]]
          .mean().round(3).to_string())


def run(stage, args, explain_one, methods):
    """Explain a range of images, appending one row per method. Resumes from what is on disk."""
    frame = images_for(args.model, args.arch, args.split)
    folder = RESULTS / stage
    folder.mkdir(parents=True, exist_ok=True)

    done = set()
    for path in result_files(stage, args.model, args.arch):
        table = pd.read_csv(path, usecols=["image", "method"])
        done |= set(zip(table.image, table.method))
    todo = frame.iloc[args.first:args.last]
    todo = todo[[not all((image, method) in done for method in methods) for image in todo.image]]
    if todo.empty:
        print(f"nothing to do: {len(frame)} images already explained by {', '.join(methods)}")
        return

    ranged = args.first > 0 or args.last is not None
    stem = f"{args.model}_{args.arch}"
    out = folder / (f"{stem}_{args.first}-{args.last or len(frame)}.csv" if ranged else f"{stem}.csv")
    device = get_device()
    batch_predict = predictor(args.model, args.arch, CHECKPOINTS, device, args.batch)
    datasets = {name: PetImages(DATA_ROOT, split=name) for name in frame.split.unique()}

    fresh = not out.is_file()
    handle = out.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=FIELDS, restval="")
    if fresh:
        writer.writeheader()
    print(f"{stage} {stem} on {device.type}: {len(todo)} images to explain with "
          f"{', '.join(methods)}, batch {args.batch} -> {out.name}", flush=True)
    started = time.perf_counter()

    for count, row in enumerate(todo.itertuples(), 1):
        tick = time.perf_counter()
        dataset = datasets[row.split]
        position = int(row.position)
        try:
            records = explain_one(dataset.array(position), dataset.mask(position), batch_predict)
        except Exception as error:  # no row is written, so the next launch retries this image
            print(f"[{count}/{len(todo)}] {row.image:32s} FAILED  {error!r}", flush=True)
            continue
        total = round(time.perf_counter() - tick, 2)
        for record in records:
            writer.writerow({"split": row.split, "position": position, "image": row.image,
                             "true_breed": row.true_breed, "species": row.species,
                             "group": row.group, "predicted_name": row.predicted_name,
                             **record, "num_samples": NUM_SAMPLES, "seed": SEED,
                             "device": device.type, "total_s": total})
        handle.flush()  # a kill costs at most the image in flight
        eta = (len(todo) - count) * (time.perf_counter() - started) / count
        print(f"[{count}/{len(todo)}] {row.image:32s} {row.group:9s} "
              + "  ".join(f"{r['method']} overlap {r['overlap']:.2f} deletion {r['deletion']:.2f}"
                          for r in records)
              + f"  {total:5.1f}s  eta {eta / 3600:5.2f}h", flush=True)
    handle.close()
