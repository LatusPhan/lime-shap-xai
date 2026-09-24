"""The ten test images the model is surest about, and the ten it is least sure about.

Same layout as figures_overlap.py, ranked by what the classifier did rather than by what
the explanation scored: the probability ResNet gave the class it chose. The top figure is
where the model is most confident, the bottom one where it is least, and each row still
shows every method's map and overlap, so the two can be read against each other.

--only wrong picks the mistakes, so the top figure becomes the confidently wrong cases,
which are usually the most revealing ones.

    python figures_accuracy.py --arch resnet50
    python figures_accuracy.py --arch resnet50 --only wrong --count 10
    python figures_accuracy.py --family                    accuracy and overlap by depth

Writes into results/figures/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import runner
from figures_overlap import COLOURS, INK, MUTED, extremes, panels, present, rows, save, style
from model import ARCHITECTURES


def family(model, archs, methods):
    """Accuracy and mean overlap per depth, two panels on one shared axis."""
    style()
    accuracy, overlap = {}, {}
    for arch in archs:
        scores = runner.scores_path(model, arch)
        if scores.is_file():
            accuracy[arch] = float(pd.read_csv(scores).correct.mean())
        try:
            frame = rows(model, arch, methods)
        except SystemExit:
            continue
        overlap[arch] = frame.groupby("method").overlap.agg(["mean", "std", "count"])
    depths = [arch for arch in archs if arch in overlap or arch in accuracy]
    if not depths:
        raise SystemExit("no depth has results yet")

    figure, axes = plt.subplots(2, 1, figsize=(1.9 * len(depths) + 3, 7), sharex=True)
    known = [arch for arch in depths if arch in accuracy]
    axes[0].bar([depths.index(a) for a in known], [accuracy[a] for a in known], 0.5, color=MUTED)
    for arch in known:
        axes[0].annotate(f"{accuracy[arch]:.1%}", (depths.index(arch), accuracy[arch]),
                         textcoords="offset points", xytext=(0, 5), ha="center",
                         fontsize=8, color=INK)
    axes[0].set_ylabel("top-1 accuracy on the test split")
    axes[0].set_title("What the model gets right", loc="left")
    axes[0].set_ylim(0, 1)

    shown = [method for method in COLOURS
             if any(method in block.index for block in overlap.values())]
    width = 0.8 / max(len(shown), 1)
    for slot, method in enumerate(shown):
        centres, means, halves = [], [], []
        for arch in depths:
            block = overlap.get(arch)
            if block is None or method not in block.index:
                continue
            centres.append(depths.index(arch) + (slot - (len(shown) - 1) / 2) * width)
            means.append(block.loc[method, "mean"])
            halves.append(1.96 * block.loc[method, "std"] / np.sqrt(block.loc[method, "count"]))
        axes[1].bar(centres, means, width * 0.88, color=COLOURS[method], label=method,
                    yerr=halves, error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 3})
        for centre, mean in zip(centres, means):
            axes[1].annotate(f"{mean:.2f}", (centre, mean), textcoords="offset points",
                             xytext=(0, 9), ha="center", fontsize=8, color=INK)
    axes[1].set_ylabel("mean evidence on the animal")
    axes[1].set_title("Where the evidence lands", loc="left")
    axes[1].set_xticks(range(len(depths)))
    axes[1].set_xticklabels(depths)
    if len(shown) > 1:
        axes[1].legend(frameon=False, ncol=len(shown), loc="upper right")
    for axis in axes:
        axis.grid(axis="y")
        axis.set_axisbelow(True)
    figure.suptitle(f"{model}: accuracy and explanation overlap by depth", x=0.01, ha="left",
                    fontsize=12)
    save(figure, f"family_{model}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("finetuned", "baseline"), default="finetuned")
    parser.add_argument("--arch", default="resnet50")
    parser.add_argument("--archs", nargs="+", default=list(ARCHITECTURES),
                        help="depths in the family figure")
    parser.add_argument("--methods", nargs="+", default=None, help="default: every method found")
    parser.add_argument("--count", type=int, default=10, help="images in each figure")
    parser.add_argument("--only", choices=("all", "right", "wrong"), default="all",
                        help="restrict to the predictions the model got right or wrong")
    parser.add_argument("--family", action="store_true", help="only the family figure")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    if args.data_root:
        runner.DATA_ROOT = Path(args.data_root)

    if args.family:
        family(args.model, args.archs, args.methods)
    else:
        frame = rows(args.model, args.arch, args.methods)
        if args.only != "all":
            frame = frame[frame.outcome == args.only]
            if frame.empty:
                raise SystemExit(f"no {args.only} predictions in the rows so far")
        score = frame.drop_duplicates("image").set_index("image").p_label
        surest, least = extremes(frame, score, args.count)
        stem = f"{args.model}_{args.arch}" + ("" if args.only == "all" else f"_{args.only}")
        kind = "" if args.only == "all" else f" {args.only} "
        panels(frame, surest, f"accuracy_surest_{stem}",
               f"{args.model} {args.arch}: the {len(surest)}{kind} images the model is surest about")
        panels(frame, least, f"accuracy_least_{stem}",
               f"{args.model} {args.arch}: the {len(least)}{kind} images the model is least sure about")
