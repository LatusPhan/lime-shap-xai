"""LIME against Kernel SHAP and Partition SHAP, on every model with stored results.

Three figures, one question each:

    comparison_metrics.png    how each method scores: overlap, deletion and insertion
    comparison_paired.png     image by image, how far each SHAP method sits from LIME.
                              The animal's area cancels within a pair, so raw overlap is
                              fair here. Runtime is a ratio, so the machine cancels too
    comparison_agreement.png  do they point at the same regions: rank correlation of the
                              region weights, and Jaccard of the top 5 regions, against LIME

Every point is a mean over the test images with its 95% interval. Only images where all
three methods explain the same class are used, so every comparison is on the same set.
Overlap leaves out images whose trimap is empty or full, as report.py does: 7 test
trimaps are empty in the dataset itself.

    python plotcomparision.py

Reads results/lime and results/shap. Writes the figures to results/figures and the
plotted numbers to results/summary/comparison.csv.
Needs only pandas and matplotlib, so it runs without torch.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # written to disk, never shown

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

RESULTS = Path(__file__).resolve().parent.parent / "results"
TOP_K = 5  # regions in the highlighted area, as in metrics.py
# the surface and method colours of figures_overlap.py, so the figures match
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dedcd6"
COLOURS = {"lime": "#2a78d6", "kernel": "#eb6834", "partition": "#1baf7a"}
MARKERS = {"lime": "o", "kernel": "s", "partition": "^"}  # second cue: the aqua is light on the surface
NAMES = {"lime": "LIME", "kernel": "Kernel SHAP", "partition": "Partition SHAP"}
METRICS = ["overlap", "deletion", "insertion"]
SHAP = ["kernel", "partition"]


def draw(table, series, panels, title, path):
    """A row of dot plots: one panel per metric, one row per model, one dot per series."""
    models = list(table.model.cat.categories)
    offsets = np.linspace(-0.22, 0.22, len(series))
    figure, axes = plt.subplots(1, len(panels), figsize=(3.7 * len(panels) + 1.2, 0.6 * len(models) + 2),
                                sharey=True, squeeze=False)
    for axis, (metric, heading, reference, limits) in zip(axes[0], panels):
        if reference is not None:
            axis.axvline(reference, color=MUTED, linewidth=1, linestyle="--")
            axis.annotate("LIME", (reference, 1), xycoords=("data", "axes fraction"), xytext=(3, -2),
                          textcoords="offset points", ha="left", va="top", fontsize=8, color=MUTED)
        for offset, method in zip(offsets, series):
            block = table[(table.metric == metric) & (table.series == method)].set_index("model").reindex(models)
            axis.errorbar(block["mean"], np.arange(len(models)) + offset,
                          xerr=[block["mean"] - block.lo, block.hi - block["mean"]],
                          fmt=MARKERS[method], color=COLOURS[method], markersize=7, elinewidth=1.6,
                          capsize=0, markeredgecolor=SURFACE, markeredgewidth=0.8)
        if limits:
            axis.set_xlim(*limits)
        axis.set_title(heading, loc="left")
        axis.grid(True, axis="x")
    axes[0][0].set_yticks(range(len(models)), models)
    axes[0][0].set_ylim(len(models) - 0.5, -0.7)  # first model on top, room for the LIME label
    handles = [Line2D([], [], color=COLOURS[m], marker=MARKERS[m], linestyle="", markersize=7,
                      label=NAMES[m]) for m in series]
    figure.legend(handles=handles, loc="upper right", frameon=False, ncol=len(handles))
    figure.suptitle(title, x=0.01, ha="left", fontsize=12)
    figure.tight_layout()  # makes room for the two-line title
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main():
    # every row, one per model, image and method
    frames = []
    for stage in ("lime", "shap"):
        for path in sorted((RESULTS / stage).glob("*.csv")):
            frame = pd.read_csv(path)
            frame["model"] = "_".join(path.stem.split("_")[:2])
            frames.append(frame)
    rows = pd.concat(frames, ignore_index=True).drop_duplicates(["model", "image", "method"])
    groups = rows.groupby(["model", "image"])
    rows = rows[groups.method.transform("nunique").eq(3) & groups.label.transform("nunique").eq(1)]
    rows = rows.assign(overlap=rows.overlap.where(rows.area.between(0, 1, inclusive="neither")))  # empty or full trimap
    models = sorted(rows.model.unique(), key=lambda m: (m.startswith("baseline"), int(m.split("resnet")[-1])))
    labels = {m: m.replace("finetuned_resnet", "ResNet-").replace("baseline_resnet50", "Stock ResNet-50")
              for m in models}

    # how each method scores
    levels = (rows.melt(id_vars=["model", "method"], value_vars=METRICS, var_name="metric")
              .groupby(["model", "method", "metric"]).value.agg(["mean", "sem", "count"]).reset_index()
              .rename(columns={"method": "series"}).assign(figure="metrics"))

    # each SHAP method minus LIME on the same image; runtime as a log ratio
    wide = rows.pivot(index=["model", "image"], columns="method", values=METRICS + ["runtime_s"])
    lime = wide.xs("lime", axis=1, level="method")
    pairs = []
    for method in SHAP:
        shap = wide.xs(method, axis=1, level="method")
        difference = shap[METRICS] - lime[METRICS]
        difference["runtime"] = np.log(shap.runtime_s / lime.runtime_s)
        pairs.append(difference.reset_index().melt(id_vars=["model", "image"], var_name="metric")
                     .assign(series=method))
    paired = (pd.concat(pairs).groupby(["model", "series", "metric"]).value.agg(["mean", "sem", "count"])
              .reset_index().assign(figure="paired"))

    # do they point at the same regions, image by image
    weights = rows.pivot(index=["model", "image"], columns="method", values="weights").dropna()
    records = []
    for (model, _), row in weights.iterrows():
        parsed = {m: np.array(row[m].split(), dtype=float) for m in ["lime"] + SHAP}
        top = {}
        for m, values in parsed.items():
            order = np.argsort(-values, kind="stable")
            top[m] = set(order[values[order] > 0][:TOP_K].tolist())  # the rule of metrics.top_positive
        for m in SHAP:
            union = top["lime"] | top[m]
            records.append({"model": model, "series": m,
                            "rank_correlation": pd.DataFrame({"a": parsed["lime"], "b": parsed[m]})
                            .corr("spearman").iat[0, 1],
                            "top5_jaccard": len(top["lime"] & top[m]) / len(union) if union else np.nan})
    agreement = (pd.DataFrame(records).melt(id_vars=["model", "series"], var_name="metric")
                 .groupby(["model", "series", "metric"]).value.agg(["mean", "sem", "count"]).reset_index()
                 .assign(figure="agreement"))

    # one table behind all three figures
    numbers = pd.concat([levels, paired, agreement], ignore_index=True)
    numbers["lo"] = numbers["mean"] - 1.96 * numbers["sem"]
    numbers["hi"] = numbers["mean"] + 1.96 * numbers["sem"]
    ratio = numbers.metric.eq("runtime")
    numbers.loc[ratio, ["mean", "lo", "hi"]] = np.exp(numbers.loc[ratio, ["mean", "lo", "hi"]])  # geometric mean
    numbers["model"] = pd.Categorical(numbers.model.map(labels), [labels[m] for m in models], ordered=True)
    numbers = numbers.sort_values(["figure", "model", "series", "metric"])
    summary, figures = RESULTS / "summary", RESULTS / "figures"
    summary.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    columns = ["figure", "model", "series", "metric", "count", "mean", "lo", "hi"]
    numbers[columns].round(4).to_csv(summary / "comparison.csv", index=False)

    # figures
    images = rows.groupby("model").image.nunique()
    count = f"{images.min():,}" if images.min() == images.max() else f"{images.min():,} to {images.max():,}"
    note = f"\nMean over {count} test images per model, with its 95% interval"
    masked = f". Overlap leaves out the {rows.loc[rows.overlap.isna(), 'image'].nunique()} with an empty animal mask"
    draw(numbers[numbers.figure == "metrics"], ["lime"] + SHAP,
         [("overlap", "Overlap: evidence on the animal\nhigher is better", None, None),
          ("deletion", "Deletion AUC\nlower is better", None, None),
          ("insertion", "Insertion AUC\nhigher is better", None, None)],
         "How LIME, Kernel SHAP and Partition SHAP score" + note + masked, figures / "comparison_metrics.png")
    draw(numbers[numbers.figure == "paired"], SHAP,
         [("overlap", "Overlap, SHAP minus LIME\nright of the line favours SHAP", 0, None),
          ("deletion", "Deletion, SHAP minus LIME\nleft of the line favours SHAP", 0, None),
          ("insertion", "Insertion, SHAP minus LIME\nright of the line favours SHAP", 0, None),
          ("runtime", "Runtime, SHAP time over LIME time\nleft of the line: SHAP is faster", 1, None)],
         "Each SHAP method against LIME on the same image" + note + masked + " (runtime: geometric mean)",
         figures / "comparison_paired.png")
    draw(numbers[numbers.figure == "agreement"], SHAP,
         [("rank_correlation", "Rank correlation of the region weights\nwith LIME's", None, (0, 1)),
          ("top5_jaccard", "Top 5 regions shared with LIME\nJaccard, 1 is the same five", None, (0, 1))],
         "Do the SHAP methods point at the same regions as LIME?" + note, figures / "comparison_agreement.png")
    print(f"wrote {summary / 'comparison.csv'} and three comparison_*.png in {figures}")


if __name__ == "__main__":
    main()
