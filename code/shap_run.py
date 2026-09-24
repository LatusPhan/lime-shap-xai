"""Stage 4: SHAP over a whole split, in the same shape as stage 3.

Two explainers, both with the 800-call budget LIME gets and the same black masking:

    kernel     KernelSHAP over LIME's own superpixels. Only the weighting of the 800
               samples differs from LIME, so a gap between them is the method itself
    partition  shap's standard image explainer. It picks its own rectangles, so its
               pixel values are summed into the same superpixels before any metric

Both explain the model's top prediction, which is the class LIME explains too, so the
two stages line up image by image. report.py checks that and drops any image where they
disagree.

    python shap_run.py --arch resnet50                        whole test split
    python shap_run.py --arch resnet50 --last 200 --seeds 0 1 2
    python shap_run.py --arch resnet152 --batch 128 --first 918 --last 1836
    python shap_run.py --status --arch resnet50
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import shap

import metrics
import runner

PARTITION_BATCH = 50  # fixed: Partition expands its tree batch by batch, so the batch changes the result
BUILT = {}            # the Partition explainer, built once per process on the first image


def explain_kernel(image, segments, label, batch_predict, seed):
    """KernelSHAP over the same superpixels: all features on is the image, all off is black."""
    np.random.seed(seed)  # KernelExplainer samples with numpy's global generator
    size = int(segments.max()) + 1
    explainer = shap.KernelExplainer(
        lambda z: metrics.masked_probability(z, image, segments, label, batch_predict),
        np.zeros((1, size)))
    # l1_reg=False keeps every superpixel; since shap 0.47 the default keeps only 10
    values = explainer.shap_values(np.ones(size), nsamples=runner.NUM_SAMPLES,
                                   l1_reg=False, silent=True)
    return np.asarray(values, dtype=float).reshape(-1), float(explainer.expected_value)


def explain_partition(image, segments, label, batch_predict):
    """Partition SHAP on pixels, summed into the superpixels the other methods use."""
    if "partition" not in BUILT:  # with a black background its base value is the same for every image
        BUILT["partition"] = shap.PartitionExplainer(
            lambda x: batch_predict(np.asarray(x, dtype=np.uint8)),
            shap.maskers.Image(0, image.shape))
    result = BUILT["partition"](image[None], max_evals=runner.NUM_SAMPLES,
                                batch_size=PARTITION_BATCH, outputs=[label], silent=True)
    pixels = np.asarray(result.values)[0].reshape(*image.shape, -1)[..., 0].sum(-1)
    weights = np.bincount(segments.ravel(), weights=pixels.ravel(),
                          minlength=int(segments.max()) + 1)
    return weights, float(np.ravel(result.base_values)[0])


def one_image(image, truth, batch_predict, seeds, batch):
    segments = metrics.superpixels(image)
    probability = batch_predict(image[None])[0]
    label = int(probability.argmax())
    rows = []

    tick = time.perf_counter()
    weights, base = explain_kernel(image, segments, label, batch_predict, seeds[0])
    runtime = time.perf_counter() - tick
    row = metrics.summarise("kernel", weights, base, runtime, image, segments, truth,
                            label, probability, batch_predict)
    row.update(metrics.stability(
        weights, [explain_kernel(image, segments, label, batch_predict, seed)[0]
                  for seed in seeds[1:]]))
    rows.append(row)

    tick = time.perf_counter()
    weights, base = explain_partition(image, segments, label, batch_predict)
    runtime = time.perf_counter() - tick
    # Partition is deterministic, so repeating it under other seeds would change nothing
    rows.append(metrics.summarise("partition", weights, base, runtime, image, segments,
                                  truth, label, probability, batch_predict))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    runner.add_common_arguments(parser)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                        help="the first gives the result, the others measure KernelSHAP stability")
    parser.add_argument("--status", action="store_true", help="how far this model has got")
    args = parser.parse_args()
    runner.apply_common(args)

    if args.status:
        runner.status("shap", args.model, args.arch, args.split)
    else:
        runner.run("shap", args,
                   lambda image, truth, predict: one_image(image, truth, predict,
                                                           args.seeds, args.batch),
                   methods=("kernel", "partition"))
