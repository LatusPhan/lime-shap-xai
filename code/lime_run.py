"""Stage 3: LIME over a whole split, one row per image, for any model.

LIME is called through the reference implementation released by its authors, with one
change: the superpixels are computed once by metrics.py and handed in, so stage 4 can
explain exactly the same regions.

    python lime_run.py --arch resnet50                       whole test split
    python lime_run.py --arch resnet50 --last 200 --seeds 0 1 2    stability on a slice
    python lime_run.py --model baseline --split both --batch 128
    python lime_run.py --arch resnet50 --first 0 --last 900   one range of four
    python lime_run.py --check                               reproduce the 24 stored records
    python lime_run.py --status --arch resnet50
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from lime import lime_image

import metrics
import runner
from data import PetImages
from model import get_device, predictor

# lime 0.2.0.1 takes no progress_bar argument, and its bar writes a line per batch
lime_image.tqdm = lambda iterable, **kwargs: iterable

RECORDS = runner.RESULTS / "lime" / "records"


def explain(image, segments, batch_predict, seed, batch, label=None):
    """One LIME explanation on the given superpixels. Returns label, weights, explanation."""
    explanation = lime_image.LimeImageExplainer(random_state=seed).explain_instance(
        image, batch_predict, top_labels=metrics.TOP_K, hide_color=0,
        num_samples=runner.NUM_SAMPLES, batch_size=batch,
        segmentation_fn=lambda _: segments, random_seed=seed)
    label = int(explanation.top_labels[0]) if label is None else label
    weights = np.zeros(int(segments.max()) + 1)
    for segment, weight in explanation.local_exp[label]:
        weights[int(segment)] = weight
    return label, weights, explanation


def one_image(image, truth, batch_predict, seeds, batch):
    segments = metrics.superpixels(image)
    probability = batch_predict(image[None])[0]
    tick = time.perf_counter()
    label, weights, explanation = explain(image, segments, batch_predict, seeds[0], batch)
    runtime = time.perf_counter() - tick

    row = metrics.summarise("lime", weights, float(explanation.intercept[label]), runtime,
                            image, segments, truth, label, probability, batch_predict)
    row["fit_r2"] = round(float(explanation.score), 6)
    row.update(metrics.stability(
        weights, [explain(image, segments, batch_predict, seed, batch, label)[1]
                  for seed in seeds[1:]]))
    return [row]


def check(batch):
    """Rerun LIME here on the stored records of the earlier grouped study and compare.

    This is the test that the new tree explains the same way the old one did: same
    superpixels, same explained class, same weights.
    """
    device = get_device()
    batch_predict = predictor("finetuned", "resnet50", runner.CHECKPOINTS, device, batch)
    dataset = PetImages(runner.DATA_ROOT, split="test")
    lookup = {dataset.name(i): i for i in range(len(dataset))}
    paths = sorted(RECORDS.glob("*.npz"))
    if not paths:
        raise SystemExit(f"no records in {RECORDS}")
    exact = 0
    for path in paths:
        stored = np.load(path, allow_pickle=True)
        meta = json.loads(str(stored["meta"]))
        image = dataset.array(lookup[meta["image"]])
        segments = metrics.superpixels(image)
        label, weights, _ = explain(image, segments, batch_predict, meta["seed"], batch)
        same = {"image": np.array_equal(image, stored["image"]),
                "segments": np.array_equal(segments, stored["segments"]),
                "label": label == meta["lime_label"]}
        gap = float(np.abs(weights - stored["weights"]).max()) if same["segments"] else float("inf")
        exact += all(same.values()) and gap < 1e-6
        print(f"{meta['image']:32s} " + "  ".join(f"{k} {v!s:5s}" for k, v in same.items())
              + f"  max weight gap {gap:.1e}", flush=True)
    print(f"{exact}/{len(paths)} stored records reproduced exactly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    runner.add_common_arguments(parser)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                        help="the first gives the result, the others measure stability")
    parser.add_argument("--check", action="store_true", help="compare with the stored records")
    parser.add_argument("--status", action="store_true", help="how far this model has got")
    args = parser.parse_args()
    runner.apply_common(args)

    if args.check:
        check(args.batch)
    elif args.status:
        runner.status("lime", args.model, args.arch, args.split)
    else:
        runner.run("lime", args,
                   lambda image, truth, predict: one_image(image, truth, predict,
                                                           args.seeds, args.batch),
                   methods=("lime",))
