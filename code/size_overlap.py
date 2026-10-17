"""Does a bigger animal get a bigger overlap? The size hypothesis, on every stored result.

Overlap is the share of the top five regions that lands on the animal, so its chance
level is the animal's own area: regions picked at random land on the animal in
proportion to its size. Raw overlap is therefore expected to rise with area even for an
explainer that knows nothing, so two hypotheses are tested, per model and method:

    H1 raw            overlap rises with animal area
    H2 beyond chance  lift rises with animal area, where lift = (overlap - area) / (1 - area)
                      is the share of the room above chance that the explanation uses:
                      0 is chance and 1 is every highlighted pixel on the animal, at any size

Both use Spearman's rho with a 95% interval. The slope of overlap on area is reported
against the chance line as well: 1 means overlap follows area exactly as chance would,
below 1 means small animals are found better than chance predicts. H2 is repeated on
right predictions only and within each species, so neither can carry the result.
Images whose mask covers none or all of the crop are left out of both tests: their
overlap is forced to 0 or 1. Seven test trimaps are empty in the dataset itself.
p values are Holm-corrected across the models and methods of each test.

    python size_overlap.py
    python size_overlap.py --trim 0.1      lift test on areas from 0.1 to 0.9 only

Reads results/lime and results/shap. Writes results/summary/size_overlap.csv, the
binned means in size_overlap_bins.csv, and results/figures/size_overlap.png.
Needs only pandas, scipy and matplotlib, so it runs without torch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # written to disk, never shown

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import linregress, spearmanr

RESULTS = Path(__file__).resolve().parent.parent / "results"
COLUMNS = ["image", "species", "group", "method", "area", "overlap"]
# the surface and method colours of figures_overlap.py, so the figures match
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dedcd6"
COLOURS = {"lime": "#2a78d6", "kernel": "#eb6834", "partition": "#1baf7a"}
MARKERS = {"lime": "o", "kernel": "s", "partition": "^"}  # second cue: the aqua is light on the surface


def rho(x, y, name):
    """Spearman's rho, its 95% interval (Bonett and Wright) and p, under one name."""
    n = len(x)
    if n < 10:
        return {f"n_{name}": n}
    r, p = spearmanr(x, y)
    half = 1.96 * np.sqrt((1 + r ** 2 / 2) / (n - 3))
    return {f"n_{name}": n, f"rho_{name}": r, f"rho_{name}_lo": np.tanh(np.arctanh(r) - half),
            f"rho_{name}_hi": np.tanh(np.arctanh(r) + half), f"p_{name}": p}


def holm(p):
    """Holm step-down adjustment; missing p values stay missing."""
    p = pd.Series(p, dtype=float)
    known = p.dropna().sort_values()
    steps = (len(known) - np.arange(len(known))) * known.to_numpy()
    p.loc[known.index] = np.minimum(1, np.maximum.accumulate(steps))
    return p


def main(args):
    # every lime and shap row, one per model, image and method
    frames = []
    for stage in ("lime", "shap"):
        for path in sorted((RESULTS / stage).glob("*.csv")):
            frame = pd.read_csv(path, usecols=lambda column: column in COLUMNS)
            if len(frame.columns) < len(COLUMNS):
                print(f"skipped {path.name}: older format without area")
                continue
            frame["model"] = "_".join(path.stem.split("_")[:2])
            frames.append(frame)
    rows = pd.concat(frames).drop_duplicates(["model", "image", "method"])
    rows = rows.dropna(subset=["area", "overlap"])
    # an empty or full mask forces the overlap, so it says nothing about size
    empty = ~rows.area.between(0, 1, inclusive="neither")
    dropped = rows[empty].image.nunique()
    rows = rows[~empty]
    rows["right"] = rows.group.isin(["correct", "exact"])
    inside = rows.area.between(args.trim, 1 - args.trim)  # lift is unstable as area nears 0 or 1
    rows["lift"] = ((rows.overlap - rows.area) / (1 - rows.area)).where(inside)
    models = sorted(rows.model.unique(),
                    key=lambda m: (m.startswith("baseline"), int(m.split("resnet")[-1])))

    # tests per model and method
    table = []
    for (model, method), block in rows.groupby(["model", "method"]):
        lifted = block.dropna(subset=["lift"])
        fit = linregress(block.area, block.overlap)
        record = {"model": model, "method": method, "images": len(block),
                  "mean_area": block.area.mean(), "mean_overlap": block.overlap.mean(),
                  "mean_lift": lifted.lift.mean(), "slope": fit.slope,
                  "slope_lo": fit.slope - 1.96 * fit.stderr, "slope_hi": fit.slope + 1.96 * fit.stderr}
        record |= rho(block.area, block.overlap, "raw")
        record |= rho(lifted.area, lifted.lift, "lift")
        right = lifted[lifted.right]
        record |= rho(right.area, right.lift, "lift_right")
        for species in ("cat", "dog"):
            group = lifted[lifted.species == species]
            record |= rho(group.area, group.lift, f"lift_{species}")
        table.append(record)
    table = pd.DataFrame(table)
    table["model"] = pd.Categorical(table.model, models, ordered=True)
    table = table.sort_values(["model", "method"]).reset_index(drop=True)
    for column in [c for c in table.columns if c.startswith("p_")]:
        table[f"{column}_holm"] = holm(table[column])

    # mean overlap and lift in area bins of 0.1, for the figure and the report
    rows["bin"] = pd.cut(rows.area, np.linspace(0, 1, 11), include_lowest=True)
    bins = (rows.groupby(["model", "method", "bin"], observed=True)
            .agg(images=("image", "size"), area=("area", "mean"),
                 overlap=("overlap", "mean"), lift=("lift", "mean"))
            .reset_index())
    bins = bins[bins.images >= args.min_bin]

    summary = RESULTS / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    table.round(6).to_csv(summary / "size_overlap.csv", index=False)
    bins.assign(bin=bins.bin.astype(str)).round(6).to_csv(summary / "size_overlap_bins.csv", index=False)

    # figure: one column per model, overlap above and lift below, chance dashed
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, len(models), figsize=(2.7 * len(models) + 1, 6.4),
                                sharex=True, sharey="row", squeeze=False)
    for column, model in enumerate(models):
        top, bottom = axes[0][column], axes[1][column]
        top.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
        bottom.axhline(0, color=MUTED, linewidth=1, linestyle="--")
        for method, line in bins[bins.model == model].groupby("method"):
            style = dict(color=COLOURS.get(method, INK), marker=MARKERS.get(method, "o"),
                         markersize=5, linewidth=1.8, markeredgecolor=SURFACE, markeredgewidth=0.8)
            top.plot(line.area, line.overlap, **style)
            bottom.plot(line.area, line.lift, **style)
        label = model.replace("finetuned_resnet", "ResNet-").replace("baseline_resnet50", "Stock ResNet-50")
        top.set_title(f"{label}, {rows[rows.model == model].image.nunique():,} images", loc="left")
        for axis in (top, bottom):
            axis.grid(True)
            axis.set_xlim(0, 1)
        bottom.set_xlabel("animal area in the crop")
    axes[0][0].set_ylim(0, 1.02)
    axes[0][0].set_ylabel("overlap: evidence on the animal")
    axes[1][0].set_ylabel("lift: share of the room above chance")
    axes[0][0].annotate("chance", (0.62, 0.55), color=MUTED, fontsize=8, rotation=38)
    axes[1][0].annotate("chance", (0.03, 0.02), color=MUTED, fontsize=8)
    handles = [Line2D([], [], color=COLOURS[m], marker=MARKERS[m], linewidth=1.8, markersize=6, label=m)
               for m in COLOURS if m in set(rows.method)]
    figure.legend(handles=handles, loc="upper right", frameon=False, ncol=len(handles))
    figure.suptitle(f"Does a bigger animal get a bigger overlap? Means in area bins of 0.1 "
                    f"with at least {args.min_bin} images", x=0.01, ha="left", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figures = RESULTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures / "size_overlap.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    shown = ["model", "method", "images", "rho_raw", "p_raw_holm", "rho_lift", "p_lift_holm",
             "rho_lift_right", "slope"]
    print(table[shown].round(4).to_string(index=False))
    print(f"{dropped} images dropped for an empty or full mask. "
          f"Lift test: areas {args.trim} to {1 - args.trim}, "
          f"{int((~inside).sum())} of {len(rows)} rows outside. "
          f"Wrote {summary / 'size_overlap.csv'} and {figures / 'size_overlap.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trim", type=float, default=0.05,
                        help="lift test only on areas from trim to 1 - trim")
    parser.add_argument("--min-bin", type=int, default=30,
                        help="fewest images for an area bin to be drawn")
    main(parser.parse_args())
