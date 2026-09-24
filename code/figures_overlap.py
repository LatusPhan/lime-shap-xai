"""The ten best and ten worst test images by overlap, with LIME and SHAP side by side.

Runs after lime_run.py and shap_run.py, from the CSVs they wrote. Each map is rebuilt
from the per-region weights stored in the row, so no image is explained again.

One row per image: the photo, then one map per method. Blue is evidence for the predicted
class, red against it. The animal is outlined in black, the top five regions in white, and
each title carries that method's overlap, so the number and the picture sit together.

    python figures_overlap.py --arch resnet50                  ten best and ten worst
    python figures_overlap.py --arch resnet50 --count 5 --methods lime kernel
    python figures_overlap.py --arch resnet50 --rank-by r2     rank by lime's surrogate fit
    python figures_overlap.py --arch resnet50 --summary        add the whole-split panels

Writes into results/figures/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are written to disk, never shown

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import metrics
import report
import runner
from data import PetImages

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dedcd6"
# fixed slots, never cycled: blue, orange, aqua. Validated together for colour blindness
COLOURS = {"lime": "#2a78d6", "kernel": "#eb6834", "partition": "#1baf7a"}
DENSITY = LinearSegmentedColormap.from_list(
    "density", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"])  # one hue, light to dark
SIGNED = LinearSegmentedColormap.from_list(
    "signed", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f2020"])  # two hues, neutral middle
FIGURES = runner.RESULTS / "figures"


def style():
    """Recessive axes and grid, ink text, one chart surface."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False})


def save(figure, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")


def rows(model, arch, methods=None):
    """Every explanation row for one model, from both stages, older files included.

    report.gather does the reading, so a file written by the earlier lime_full.py is
    brought into the same schema here as it is in the summaries.
    """
    frame = report.gather(model, arch)
    if frame is None:
        raise SystemExit(f"no rows for {model}/{arch}. Run lime_run.py and shap_run.py first")
    if methods:
        frame = frame[frame.method.isin(methods)]
    # a frame with no lime rows has an empty fit column, which must still read as a number
    frame["fit_r2"] = pd.to_numeric(frame.fit_r2, errors="coerce")
    frame["outcome"] = np.where(frame.group.isin(("correct", "exact")), "right", "wrong")
    return frame


def present(frame):
    """The methods in the data, in the fixed colour order."""
    return [method for method in COLOURS if method in set(frame.method)]


def extremes(frame, score, count):
    """The highest and lowest scoring images that every method has kept weights for."""
    drawable = frame[frame.weights.notna()]
    complete = drawable.pivot(index="image", columns="method", values="overlap").dropna().index
    score = score[score.index.isin(complete)].dropna().sort_values()
    if score.empty:
        raise SystemExit("no image has every method explained with its weights kept. "
                         "Rows from the earlier lime_full.py carry no weights.")
    count = min(count, len(score))
    return list(score.index[-count:][::-1]), list(score.index[:count])


def draw(axis, image, segments, weights, truth, title, colour):
    """The photo under its signed map: blue is evidence for the class, red against it."""
    attribution = weights[segments]
    limit = float(np.abs(attribution).max()) or 1.0
    axis.imshow(image)
    axis.imshow(attribution, cmap=SIGNED, vmin=-limit, vmax=limit, alpha=0.55)
    axis.contour(truth.astype(float), levels=[0.5], colors=INK, linewidths=1.0)
    axis.contour(np.isin(segments, metrics.top_positive(weights)).astype(float),
                 levels=[0.5], colors="#ffffff", linewidths=1.2)
    axis.set_title(title, fontsize=8, loc="left", color=colour)
    axis.axis("off")


def panels(frame, images, name, note):
    """One row per image: the photo, then one map per method."""
    style()
    methods = present(frame)
    datasets = {split: PetImages(runner.DATA_ROOT, split=split) for split in frame.split.unique()}
    figure, axes = plt.subplots(len(images), len(methods) + 1,
                               figsize=(2.7 * (len(methods) + 1), 2.9 * len(images)), squeeze=False)
    for index, image_name in enumerate(images):
        block = frame[frame.image == image_name].set_index("method")
        first = block.iloc[0]
        dataset = datasets[first.split]
        picture = dataset.array(int(first.position))
        truth = dataset.mask(int(first.position))
        segments = metrics.superpixels(picture)

        axis = axes[index][0]
        axis.imshow(picture)
        axis.contour(truth.astype(float), levels=[0.5], colors=INK, linewidths=1.0)
        axis.set_title(f"{image_name}\ntrue {first.true_breed}, said {first.predicted_name} "
                       f"({first.p_label:.2f})\nanimal area {first.area:.2f}, model {first.outcome}",
                       fontsize=8, loc="left")
        axis.axis("off")
        for column, method in enumerate(methods, start=1):
            record = block.loc[method]
            fit = f"  R2 {record.fit_r2:.2f}" if np.isfinite(record.fit_r2) else ""
            draw(axes[index][column], picture, segments,
                 np.array(str(record.weights).split(), dtype=float), truth,
                 f"{method}  overlap {record.overlap:.2f}{fit}", COLOURS[method])
    figure.suptitle(note, x=0.01, ha="left", fontsize=12)
    save(figure, f"{name}.png")


def distribution(axis, frame, column, title, label):
    """One box per method and outcome. Medians are labelled, which is also the aqua relief."""
    methods = present(frame)
    ticks, centres = [], []
    for block, outcome in enumerate(("right", "wrong")):
        for slot, method in enumerate(methods):
            values = frame[(frame.method == method) & (frame.outcome == outcome)][column].dropna()
            if values.empty:
                continue
            position = block * (len(methods) + 1) + slot
            box = axis.boxplot([values], positions=[position], widths=0.7, patch_artist=True,
                               showfliers=False, medianprops={"color": INK, "linewidth": 1.4},
                               whiskerprops={"color": GRID}, capprops={"color": GRID},
                               boxprops={"edgecolor": SURFACE, "linewidth": 2})
            box["boxes"][0].set_facecolor(COLOURS[method])
            axis.annotate(f"{values.median():.2f}", (position, values.median()),
                          textcoords="offset points", xytext=(0, 7), ha="center",
                          fontsize=8, color=INK)
        centres.append(block * (len(methods) + 1) + (len(methods) - 1) / 2)
        ticks.append(f"model {outcome}\nn={frame[(frame.outcome == outcome) & (frame.method == methods[0])].shape[0]:,}")
    axis.set_xticks(centres)
    axis.set_xticklabels(ticks)
    axis.set_ylabel(label)
    axis.set_title(title, loc="left")
    axis.grid(axis="y")
    axis.set_axisbelow(True)


def bars(axis, frame, columns, title, label):
    """Mean per method for each column, with 95 per cent intervals and direct labels."""
    methods = present(frame)
    width = 0.8 / len(methods)
    for slot, method in enumerate(methods):
        block = frame[frame.method == method]
        means = [block[column].mean() for column in columns]
        halves = [1.96 * block[column].std(ddof=1) / np.sqrt(block[column].notna().sum())
                  for column in columns]
        centres = np.arange(len(columns)) + (slot - (len(methods) - 1) / 2) * width
        axis.bar(centres, means, width * 0.88, color=COLOURS[method], label=method,
                 yerr=halves, error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 3})
        for centre, mean in zip(centres, means):
            axis.annotate(f"{mean:.2f}", (centre, mean), textcoords="offset points",
                          xytext=(0, 9), ha="center", fontsize=8, color=INK)
    axis.set_xticks(np.arange(len(columns)))
    axis.set_xticklabels(columns)
    axis.set_ylabel(label)
    axis.set_title(title, loc="left")
    axis.grid(axis="y")
    axis.set_axisbelow(True)


def summary(frame, model, arch):
    """The whole split behind the examples: distributions, faithfulness, cost, and the area."""
    style()
    methods = present(frame)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    distribution(axes[0][0], frame, "overlap", "Evidence on the animal", "share of the top 5 regions")
    distribution(axes[0][1], frame, "excess", "After removing the animal's own area",
                 "overlap minus area")
    if frame.deletion.notna().any():
        bars(axes[1][0], frame, ["deletion", "insertion"],
             "Faithfulness: hide the top regions, then reveal them", "area under the curve")
        axes[1][0].set_xticklabels(["deletion\nlower is better", "insertion\nhigher is better"])
    else:
        axes[1][0].set_visible(False)
    block = frame.dropna(subset=["area", "overlap"])
    mesh = axes[1][1].hexbin(block.area, block.overlap, gridsize=34, cmap=DENSITY,
                             mincnt=1, linewidths=0)
    axes[1][1].plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
    axes[1][1].annotate(f"{float((block.overlap > block.area).mean()):.0%} above the chance line",
                        (0.04, 0.95), xycoords="axes fraction", fontsize=8, color=INK)
    axes[1][1].set_xlabel("animal area in the crop")
    axes[1][1].set_ylabel("evidence on the animal")
    axes[1][1].set_title("Overlap against the area it is a share of", loc="left")
    figure.colorbar(mesh, ax=axes[1][1], label="images")

    handles = [Line2D([], [], marker="s", linestyle="", markersize=9, color=COLOURS[m], label=m)
               for m in methods]
    figure.legend(handles=handles, loc="upper right", frameon=False, ncol=len(methods))
    figure.suptitle(f"{model} {arch}: {frame.image.nunique():,} test images, "
                    f"{', '.join(methods)}", x=0.01, ha="left", fontsize=12)
    save(figure, f"summary_{model}_{arch}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("finetuned", "baseline"), default="finetuned")
    parser.add_argument("--arch", default="resnet50")
    parser.add_argument("--methods", nargs="+", default=None, help="default: every method found")
    parser.add_argument("--count", type=int, default=10, help="images in each figure")
    parser.add_argument("--rank-by", choices=("overlap", "r2"), default="overlap",
                        help="overlap averaged over the methods, or lime's surrogate fit")
    parser.add_argument("--summary", action="store_true", help="also draw the whole-split panels")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    if args.data_root:
        runner.DATA_ROOT = Path(args.data_root)

    frame = rows(args.model, args.arch, args.methods)
    if args.rank_by == "r2":
        score = frame[frame.method == "lime"].set_index("image").fit_r2
        measure = "lime's surrogate fit"
    else:
        score = frame.pivot(index="image", columns="method", values="overlap").mean(axis=1)
        measure = "overlap, averaged over the methods"
    best, worst = extremes(frame, score, args.count)
    stem = f"{args.model}_{args.arch}"
    panels(frame, best, f"overlap_best_{stem}",
           f"{args.model} {args.arch}: the {len(best)} highest images by {measure}")
    panels(frame, worst, f"overlap_worst_{stem}",
           f"{args.model} {args.arch}: the {len(worst)} lowest images by {measure}")
    if args.summary:
        summary(frame, args.model, args.arch)
