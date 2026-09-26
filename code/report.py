"""Stage 5: turn the per-image rows into the numbers a report can quote.

Three kinds of comparison, each with the column that is fair for it:

    between methods   paired image by image, so the animal's area cancels and raw
                      overlap is the honest column
    right vs wrong    between different images, so the area-corrected column is the
                      honest one, and both are reported
    agreement         rank correlation and top-5 overlap between two methods' weights

Files written by the earlier lime_full.py are read too. They carry no weights and no
curves, so they only contribute overlap.

    python report.py                      every model found in results/
    python report.py --arch resnet50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

import metrics
import runner
from data import PetImages
from model import ARCHITECTURES

LEGACY = {"object_overlap": "overlap", "local_fit_r2": "fit_r2", "lime_label": "label",
          "predicted_probability": "p_label"}
COLUMNS = ("overlap", "excess", "deletion", "insertion", "additivity", "runtime_s")
PAIRED = ("overlap", "deletion", "insertion", "runtime_s")


def describe(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"n": 0}
    half = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values)) if len(values) > 1 else float("nan")
    return {"n": len(values), "mean": round(float(values.mean()), 4),
            "median": round(float(np.median(values)), 4),
            "std": round(float(values.std(ddof=1)), 4) if len(values) > 1 else None,
            "ci95": [round(float(values.mean() - half), 4), round(float(values.mean() + half), 4)],
            "min": round(float(values.min()), 4), "max": round(float(values.max()), 4)}


def contrast(right, wrong):
    """One-sided Mann-Whitney: does the metric sit higher when the model is right?"""
    right = np.asarray(right, dtype=float)
    wrong = np.asarray(wrong, dtype=float)
    right, wrong = right[~np.isnan(right)], wrong[~np.isnan(wrong)]
    if len(right) < 2 or len(wrong) < 2:
        return {"n_right": len(right), "n_wrong": len(wrong)}
    statistic, p = mannwhitneyu(right, wrong, alternative="greater")
    pairs = len(right) * len(wrong)
    return {"n_right": len(right), "n_wrong": len(wrong),
            "right_mean": round(float(right.mean()), 4),
            "wrong_mean": round(float(wrong.mean()), 4),
            "gap": round(float(right.mean() - wrong.mean()), 4),
            "p_one_sided": float(p),
            "rank_biserial_r": round(2 * float(statistic) / pairs - 1, 4)}


def gather(model, arch):
    """Every row for one model, from both stages, with older files brought into the schema."""
    frames = []
    for stage in ("lime", "shap"):
        for path in runner.result_files(stage, model, arch):
            frame = pd.read_csv(path).rename(columns=LEGACY)
            if "method" not in frame.columns:  # written by the earlier lime_full.py
                frame["method"] = "lime"
            frames.append(frame)
    if not frames:
        return None
    frame = pd.concat(frames, ignore_index=True).drop_duplicates(["image", "method"])
    for column in runner.FIELDS:  # older files carry fewer columns; the rest stay empty
        if column not in frame.columns:
            frame[column] = np.nan

    # older rows have no animal area, and the corrected metric needs it
    blank = frame.area.isna()
    if blank.any():
        print(f"  reading trimaps for {int(blank.sum())} rows with no animal area")
        datasets = {split: PetImages(runner.DATA_ROOT, split=split)
                    for split in frame.loc[blank, "split"].unique()}
        cache = {}
        for row in frame[blank].itertuples():
            key = (row.split, row.image)
            if key not in cache:
                cache[key] = float(datasets[row.split].mask(int(row.position)).mean())
        frame.loc[blank, "area"] = [cache[(r.split, r.image)] for r in frame[blank].itertuples()]
        frame.loc[blank, "excess"] = frame.loc[blank, "overlap"] - frame.loc[blank, "area"]
    return frame.reset_index(drop=True)


def expected_count(model, arch, splits):
    """How many images the explained split holds, when the predictions table is at hand."""
    try:
        return len(runner.images_for(model, arch, splits[0] if len(splits) == 1 else "both"))
    except SystemExit:
        return None


def summarise(model, arch, frame):
    labels = frame.pivot(index="image", columns="method", values="label")
    agreed = labels.apply(lambda row: row.dropna().nunique() <= 1, axis=1)
    if not agreed.all():
        print(f"  {int((~agreed).sum())} images explain different classes in different "
              "stages and are left out of the paired parts")
    shared = set(agreed[agreed].index)

    summary = {"model": model, "arch": arch,
               "images": int(frame.image.nunique()),
               "expected": expected_count(model, arch, frame.split.dropna().unique()),
               "devices": sorted(frame.device.dropna().astype(str).unique()),
               "methods": {}, "right_vs_wrong": {}, "paired": {}, "agreement": {}, "stability": {}}

    for method, block in frame.groupby("method"):
        summary["methods"][method] = {c: describe(block[c]) for c in COLUMNS}
        right = block.group.isin(("correct", "exact")).to_numpy()
        summary["right_vs_wrong"][method] = {
            c: contrast(block[c].to_numpy()[right], block[c].to_numpy()[~right])
            for c in ("overlap", "excess")}
        stability = {c: describe(block[c]) for c in ("seed_rho", "seed_jac")}
        if stability["seed_rho"]["n"]:
            summary["stability"][method] = stability

    methods = sorted(frame.method.unique())
    pairs = [(a, b) for i, a in enumerate(methods) for b in methods[i + 1:]]
    for column in PAIRED:
        wide = frame[frame.image.isin(shared)].pivot(index="image", columns="method", values=column)
        for a, b in pairs:
            pair = wide[[a, b]].dropna()
            difference = pair[a] - pair[b]
            entry = {"n": len(pair), "median_difference": round(float(difference.median()), 4)}
            if len(pair) >= 10 and difference.any():
                entry["wilcoxon_p"] = float(wilcoxon(pair[a], pair[b]).pvalue)
            summary["paired"][f"{column}_{a}_vs_{b}"] = entry

    weights = {method: {row.image: np.array(row.weights.split(), dtype=float)
                        for row in block.itertuples() if isinstance(row.weights, str)}
               for method, block in frame.groupby("method")}
    for a, b in pairs:
        common = sorted(set(weights[a]) & set(weights[b]) & shared)
        if not common:
            continue
        rho = [spearmanr(weights[a][image], weights[b][image])[0] for image in common]
        jac = [metrics.jaccard(metrics.top_positive(weights[a][image]),
                               metrics.top_positive(weights[b][image])) for image in common]
        summary["agreement"][f"{a}_vs_{b}"] = {"rank_correlation": describe(rho),
                                               "top5_jaccard": describe(jac)}
    return summary


def main(archs, models):
    runner.RESULTS.joinpath("summary").mkdir(parents=True, exist_ok=True)
    table = []
    for model in models:
        for arch in (["resnet50"] if model == "baseline" else archs):
            frame = gather(model, arch)
            if frame is None:
                continue
            print(f"{model}/{arch}: {frame.image.nunique()} images, "
                  f"{', '.join(sorted(frame.method.unique()))}")
            summary = summarise(model, arch, frame)
            path = runner.RESULTS / "summary" / f"{model}_{arch}.json"
            path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            for method, block in summary["methods"].items():
                table.append({"model": f"{model}_{arch}", "method": method,
                              "images": block["overlap"]["n"],
                              **{c: block[c].get("mean") for c in
                                 ("overlap", "excess", "deletion", "insertion", "additivity")},
                              "median_runtime_s": block["runtime_s"].get("median"),
                              "right_vs_wrong_p": summary["right_vs_wrong"][method]["excess"]
                              .get("p_one_sided")})
    if not table:
        raise SystemExit(f"no results under {runner.RESULTS}")
    table = pd.DataFrame(table)
    table.to_csv(runner.RESULTS / "summary" / "summary_table.csv", index=False)
    print()
    print(table.to_string(index=False))
    print(f"\nwrote {runner.RESULTS / 'summary'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archs", nargs="+", default=list(ARCHITECTURES))
    parser.add_argument("--models", nargs="+", default=["finetuned", "baseline"],
                        choices=("finetuned", "baseline"))
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    if args.data_root:
        runner.DATA_ROOT = Path(args.data_root)
    main(args.archs, args.models)
