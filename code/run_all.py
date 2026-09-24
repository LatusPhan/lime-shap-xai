"""The whole pipeline from one command, on macOS, Linux or Windows.

Each stage is also a script you can run on its own; this only calls them in order and,
where it helps, several at once. The explainers are limited by the processor rather than
the GPU, so --parallel 4 on one GPU is close to four times the throughput.

    python run_all.py download
    python run_all.py train --archs resnet50
    python run_all.py score --archs resnet50 resnet101
    python run_all.py explain --archs resnet50 --parallel 4 --batch 128
    python run_all.py report
    python run_all.py all --archs resnet18 resnet34 resnet50 resnet101 resnet152 --parallel 4

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
    LOGS.mkdir(parents=True, exist_ok=True)

    def start(job):
        name, command = job
        print("  " + " ".join(command[1:]), flush=True)
        if parallel == 1:
            return subprocess.run(command).returncode
        with (LOGS / f"{name}.log").open("w", encoding="utf-8") as log:
            return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        codes = list(pool.map(start, jobs))
    failed = [name for (name, _), code in zip(jobs, codes) if code]
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}. See results/logs")


def stage(name, args, extra):
    """One command line for a stage script, with the shared arguments filled in."""
    command = [sys.executable, "-u", str(CODE / f"{name}.py"),
               "--model", args.model, "--batch", str(args.batch)]
    if args.data_root:
        command += ["--data-root", args.data_root]
    return command + extra


def explain(args, arch):
    """The ranges for one model: a stability slice first, then the split in equal parts."""
    split = "both" if args.model == "baseline" else "test"
    total = len(runner.images_for(args.model, arch, split))
    jobs = []
    for script in ("lime_run", "shap_run"):
        shared = ["--arch", arch, "--split", split]
        if args.stability:
            jobs.append((f"{script}_{arch}_stability",
                         stage(script, args, shared + ["--last", str(args.stability),
                                                       "--seeds", "0", "1", "2"])))
        if args.parallel == 1:
            jobs.append((f"{script}_{arch}", stage(script, args, shared)))
            continue
        step = -(-total // args.parallel)  # ceiling, so the last range reaches the end
        for start in range(0, total, step):
            jobs.append((f"{script}_{arch}_{start}",
                         stage(script, args, shared + ["--first", str(start),
                                                       "--last", str(min(start + step, total))])))
    return jobs


def main(args):
    if args.data_root:
        runner.DATA_ROOT = Path(args.data_root)
    archs = ["resnet50"] if args.model == "baseline" else args.archs

    if args.action in ("download", "all"):
        download(runner.DATA_ROOT)
    if args.action in ("train", "all") and args.model == "finetuned":
        launch([(f"train_{arch}", [sys.executable, "-u", str(CODE / "train.py"), "--arch", arch,
                                   "--epochs", str(args.epochs), "--workers", str(args.workers)]
                 + (["--data-root", args.data_root] if args.data_root else []))
                for arch in archs], parallel=1)  # one at a time: training wants the whole GPU
    if args.action in ("score", "all"):
        launch([(f"score_{arch}", stage("score", args, ["--arch", arch,
                                                        "--split", "both" if args.model == "baseline" else "test"]))
                for arch in archs], parallel=1)
    if args.action in ("explain", "all"):
        for arch in archs:
            # the stability slice must finish before the ranges, so they skip those images
            jobs = explain(args, arch)
            stability = [job for job in jobs if job[0].endswith("stability")]
            launch(stability, parallel=min(len(stability), args.parallel) or 1)
            launch([job for job in jobs if not job[0].endswith("stability")], args.parallel)
    if args.action in ("report", "all"):
        launch([("report", [sys.executable, "-u", str(CODE / "report.py"), "--models", args.model,
                            "--archs", *archs]
                 + (["--data-root", args.data_root] if args.data_root else []))], parallel=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("download", "train", "score", "explain", "report", "all"))
    parser.add_argument("--model", choices=("finetuned", "baseline"), default="finetuned")
    parser.add_argument("--archs", nargs="+", default=["resnet50"], choices=list(ARCHITECTURES))
    parser.add_argument("--batch", type=int, default=32, help="images per forward pass; 128 suits a GPU")
    parser.add_argument("--parallel", type=int, default=1,
                        help="ranges to explain at once; 4 suits a GPU with 4 spare cores")
    parser.add_argument("--stability", type=int, default=0,
                        help="images to explain under three seeds before the rest, 0 to skip")
    parser.add_argument("--epochs", type=int, default=10, help="train only")
    parser.add_argument("--workers", type=int, default=4, help="train only; 0 on Windows if loaders hang")
    parser.add_argument("--data-root", default=None)
    main(parser.parse_args())
