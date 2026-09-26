"""The whole pipeline from one command, on macOS, Linux or Windows.

Each stage is also a script you can run on its own; this only calls them in order and,
where it helps, several at once. The explainers are limited by the processor rather than
the GPU, so --parallel 4 on one GPU is close to four times the throughput.

    python run_all.py download
    python run_all.py train   --archs resnet50 --retrain
    python run_all.py score   --archs resnet50 resnet101
    python run_all.py explain --archs resnet50 --parallel 4 --batch 128
    python run_all.py report  --models finetuned baseline

Every model, LIME and both SHAP methods over the test split, then summaries and figures:

    python run_all.py all --models finetuned baseline --archs resnet50 resnet18 resnet34 resnet101 resnet152 --batch 128 --parallel 4 --stability 200

Nothing is done twice. A checkpoint is kept unless --retrain, a predictions table unless
--rescore, and the explainers resume image by image, so the same command picks up where
a stopped run left off. Models run in the order given.

Logs from parallel ranges go to results/logs, one file per range, because four processes
writing to one console is unreadable.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import runner
from data import download
from model import ARCHITECTURES

CODE = Path(__file__).resolve().parent
LOGS = runner.RESULTS / "logs"


def launch(jobs, parallel):
    """Run (name, command) jobs, at most `parallel` at a time. Stops if one fails."""
    if not jobs:
        return
    LOGS.mkdir(parents=True, exist_ok=True)

    def start(job):
        name, command = job
        print("  " + " ".join(command[1:]), flush=True)
        if parallel == 1:
            return subprocess.run(command).returncode
        # appended, so a restart keeps the reason an earlier attempt stopped
        with (LOGS / f"{name}.log").open("a", encoding="utf-8") as log:
            return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        codes = list(pool.map(start, jobs))
    failed = [name for (name, _), code in zip(jobs, codes) if code]
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}. See results/logs")


def stage(name, model, args, extra):
    """One command line for a stage script, with the shared arguments filled in."""
    command = [sys.executable, "-u", str(CODE / f"{name}.py"),
               "--model", model, "--batch", str(args.batch)]
    if args.data_root:
        command += ["--data-root", args.data_root]
    return command + extra


def explain(model, arch, args):
    """The ranges for one model: a stability slice first, then the split in equal parts."""
    split = args.split if model == "baseline" else "test"
    total = len(runner.images_for(model, arch, split))
    jobs = []
    for name in ("lime_run", "shap_run"):
        shared = ["--arch", arch, "--split", split]
        label = f"{name}_{model}_{arch}"
        if args.stability:
            jobs.append((f"{label}_stability",
                         stage(name, model, args, shared + ["--last", str(args.stability),
                                                            "--seeds", "0", "1", "2"])))
        if args.parallel == 1:
            jobs.append((label, stage(name, model, args, shared)))
            continue
        step = -(-total // args.parallel)  # ceiling, so the last range reaches the end
        for start in range(0, total, step):
            jobs.append((f"{label}_{start}",
                         stage(name, model, args, shared + ["--first", str(start),
                                                            "--last", str(min(start + step, total))])))
    return jobs


def main(args):
    if args.data_root:
        runner.DATA_ROOT = Path(args.data_root)
    # the baseline is the stock ImageNet ResNet-50, so it has one depth only
    runs = [(model, arch) for model in args.models
            for arch in (["resnet50"] if model == "baseline" else args.archs)]
    depths = [arch for model, arch in runs if model == "finetuned"]
    root = ["--data-root", args.data_root] if args.data_root else []

    if args.action in ("download", "all"):
        download(runner.DATA_ROOT)

    retrained = []
    if args.action in ("train", "all"):
        retrained = [arch for arch in depths if args.retrain
                     or not (runner.CHECKPOINTS / f"{arch}_pets.pt").is_file()]
        # new weights under old explanations would put two models in one table
        stale = [arch for arch in retrained if runner.result_files("lime", "finetuned", arch)
                 or runner.result_files("shap", "finetuned", arch)]
        if stale:
            raise SystemExit(f"results/lime or results/shap already explain the current "
                             f"{', '.join(stale)} weights; move those files before retraining")
        # one at a time: training wants the whole GPU
        launch([(f"train_{arch}", [sys.executable, "-u", str(CODE / "train.py"), "--arch", arch,
                                   "--epochs", str(args.epochs), "--workers", str(args.workers)]
                 + root) for arch in retrained], parallel=1)

    if args.action in ("score", "explain", "all") or retrained:
        # predictions come before explanations; a table is redone only for new weights
        todo = [(model, arch) for model, arch in runs
                if args.rescore or (model == "finetuned" and arch in retrained)
                or not runner.scores_path(model, arch).is_file()]
        launch([(f"score_{model}_{arch}",
                 stage("score", model, args,
                       ["--arch", arch, "--split", "both" if model == "baseline" else "test"]))
                for model, arch in todo], parallel=1)

    if args.action in ("explain", "all"):
        for model, arch in runs:
            # the stability slice must finish before the ranges, so they skip those images
            jobs = explain(model, arch, args)
            stability = [job for job in jobs if job[0].endswith("stability")]
            launch(stability, parallel=min(len(stability), args.parallel))
            launch([job for job in jobs if not job[0].endswith("stability")], args.parallel)

    if args.action in ("report", "all"):
        launch([("report", [sys.executable, "-u", str(CODE / "report.py"),
                            "--models", *args.models, "--archs", *args.archs] + root)], parallel=1)
        # figures for every model with rows so far, and the depth comparison when there are several
        drawn = [(model, arch) for model, arch in runs if runner.result_files("lime", model, arch)
                 or runner.result_files("shap", model, arch)]
        figures = [(f"{name}_{model}_{arch}", [sys.executable, "-u", str(CODE / f"{name}.py"),
                                               "--model", model, "--arch", arch] + root)
                   for model, arch in drawn for name in ("figures_overlap", "figures_accuracy")]
        family = [arch for model, arch in drawn if model == "finetuned"]
        if len(family) > 1:
            figures.append(("figures_family", [sys.executable, "-u", str(CODE / "figures_accuracy.py"),
                                               "--family", "--archs", *family] + root))
        launch(figures, parallel=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("download", "train", "score", "explain", "report", "all"))
    parser.add_argument("--models", "--model", dest="models", nargs="+", default=["finetuned"],
                        choices=("finetuned", "baseline"),
                        help="finetuned, baseline or both, run in the order given")
    parser.add_argument("--archs", nargs="+", default=["resnet50"], choices=list(ARCHITECTURES),
                        help="fine-tuned depths, run in the order given")
    parser.add_argument("--split", choices=("test", "both"), default="test",
                        help="images the baseline explains; the fine-tuned models always use test")
    parser.add_argument("--batch", type=int, default=32, help="images per forward pass; 128 suits a GPU")
    parser.add_argument("--parallel", type=int, default=1,
                        help="ranges to explain at once; 4 suits a GPU with 4 spare cores")
    parser.add_argument("--stability", type=int, default=0,
                        help="images to explain under three seeds before the rest, 0 to skip")
    parser.add_argument("--retrain", action="store_true",
                        help="train even when a checkpoint exists; its predictions are redone too")
    parser.add_argument("--rescore", action="store_true", help="redo existing predictions tables")
    parser.add_argument("--epochs", type=int, default=10, help="train only")
    parser.add_argument("--workers", type=int, default=4, help="train only; 0 on Windows if loaders hang")
    parser.add_argument("--data-root", default=None)
    main(parser.parse_args())
