"""Everything measured about an explanation, so every method is measured the same way.

An explanation here is one signed weight per superpixel. The superpixels come from
lime_image's own quickshift settings, and every method is scored on them, so the
numbers compare directly.
"""

from __future__ import annotations

import numpy as np
from lime.wrappers.scikit_image import SegmentationAlgorithm
from scipy.stats import spearmanr

TOP_K = 5  # regions in the highlighted area, as in lime's get_image_and_mask
SEGMENTER = SegmentationAlgorithm("quickshift", kernel_size=4, max_dist=200, ratio=0.2,
                                  random_seed=0)  # lime_image's default settings


def superpixels(image):
    """The regions every method explains. Deterministic, so two runs always agree."""
    return SEGMENTER(image)


def masked_probability(z, image, segments, label, batch_predict):
    """p(label) for each coalition row. A 0 paints that superpixel black, like hide_color=0."""
    keep = (np.asarray(z) > 0.5)[:, segments]  # boolean before indexing: 40 MB at 800 rows, not 320
    return batch_predict(np.where(keep[..., None], image, np.uint8(0)))[:, label]


def top_positive(weights):
    """The TOP_K largest positive weights, the rule of lime's get_image_and_mask."""
    order = np.argsort(-weights, kind="stable")
    return order[weights[order] > 0][:TOP_K]


def jaccard(a, b):
    a, b = set(np.asarray(a).tolist()), set(np.asarray(b).tolist())
    return len(a & b) / len(a | b) if a | b else float("nan")


def object_overlap(weights, segments, truth):
    """Share of the highlighted region that falls on the animal.

    1.0 means the explanation pointed entirely at the animal, 0.0 entirely at the
    background. Its chance level is the animal's own area, which is why `excess`
    reports the difference rather than the raw value alone.
    """
    region = np.isin(segments, top_positive(weights))
    highlighted = int(region.sum())
    if highlighted == 0:
        return float("nan")
    return float((region & truth).sum() / highlighted)


def curves(weights, image, segments, label, batch_predict):
    """Deletion and insertion AUC: superpixels hidden or revealed most positive first.

    x is the share of pixels touched, so regions of different sizes are comparable.
    A faithful explanation deletes fast (low) and inserts fast (high).
    """
    order = np.argsort(-weights, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    kept = (rank[None, :] >= np.arange(len(order) + 1)[:, None]).astype(float)  # row k hides the top k
    share = np.concatenate([[0.0], np.cumsum(np.bincount(segments.ravel(), minlength=len(order))[order])])
    share /= segments.size
    deletion = np.trapezoid(masked_probability(kept, image, segments, label, batch_predict), share)
    insertion = np.trapezoid(masked_probability(1 - kept, image, segments, label, batch_predict), share)
    return float(deletion), float(insertion)


def summarise(method, weights, base, runtime, image, segments, truth, label, probability,
              batch_predict):
    """The metric columns every method reports. `base` is the explanation's own intercept."""
    area = float(truth.mean())
    overlap = object_overlap(weights, segments, truth)
    deletion, insertion = curves(weights, image, segments, label, batch_predict)
    target = float(probability[label])
    return {"method": method, "label": int(label), "p_label": round(target, 6),
            "num_segments": len(weights), "area": round(area, 6),
            "overlap": round(overlap, 6), "excess": round(overlap - area, 6),
            "deletion": round(deletion, 6), "insertion": round(insertion, 6),
            "additivity": round(abs(float(weights.sum()) + base - target), 6),
            "runtime_s": round(runtime, 2),
            "weights": " ".join(f"{v:.6g}" for v in weights)}


def stability(weights, repeats):
    """Agreement between the first run and the repeats under other seeds."""
    if not repeats:
        return {}
    return {"seed_rho": round(float(np.mean([spearmanr(weights, r)[0] for r in repeats])), 6),
            "seed_jac": round(float(np.mean([jaccard(top_positive(weights), top_positive(r))
                                             for r in repeats])), 6)}
