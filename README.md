# xai: LIME and SHAP on the same pet classifier

Code and results are separate. Everything under `results/` was produced by running the
code in `code/` on a machine of mine. Nothing here is copied from a paper.

```
xai/
  code/        one file per stage, plus three shared modules
  results/     checkpoints, scores, explanations, summaries
  data/        Oxford-IIIT Pet, downloaded per machine (not copied between them)
```

## The stages

| File | What it does | Writes |
|---|---|---|
| `train.py` | Fine-tunes one ResNet depth. Best epoch picked on a validation slice of trainval. | `results/checkpoints/<arch>_pets.pt` |
| `score.py` | What the model predicts for every image, before any explanation. | `results/scores/<model>.csv` |
| `lime_run.py` | LIME over a whole split, one row per image. | `results/lime/<model>_<arch>.csv` |
| `shap_run.py` | KernelSHAP and Partition SHAP, same images, same budget, same metrics. | `results/shap/<model>_<arch>.csv` |
| `report.py` | Per-method summaries, right against wrong, and the paired method comparison. | `results/summary/` |
| `run_all.py` | Runs the stages in order, and several ranges at once. | `results/logs/` |
| `figures_overlap.py` | Ten best and ten worst images by overlap, every method side by side. | `results/figures/` |
| `figures_accuracy.py` | Ten images the model is surest about and ten it is least sure about. | `results/figures/` |

Shared modules: `data.py` (dataset, transforms, animal masks), `model.py` (build, load,
`batch_predict`, device), `metrics.py` (superpixels, overlap, curves, stability),
`runner.py` (the loop every explainer shares: image order, resume, ranges, rows).

## What is measured

Every method produces one signed weight per superpixel, and is then scored the same way:

- **overlap**: share of the top five positive superpixels that lands on the animal.
- **excess**: overlap minus the animal's own area, which is the metric's chance level.
- **deletion / insertion**: area under the curve as superpixels are blacked out or
  revealed, most positive first, with the share of pixels on the x axis.
- **additivity**: how far the weights plus the intercept sit from the prediction. Zero by
  construction for both SHAP methods, not for LIME.
- **runtime** and, with extra `--seeds`, **stability** across seeds.

All three methods get 800 model calls per image and black masking, so the comparison is
about the method rather than the budget.

## Running it

Every model, LIME and both SHAP methods over the whole test split, then the summaries
and the figures. The same line works in PowerShell, zsh and bash:

```bash
python code/run_all.py all --models finetuned baseline --archs resnet50 resnet18 resnet34 resnet101 resnet152 --batch 128 --parallel 4 --stability 200
```

Nothing is done twice. Checkpoints are kept unless `--retrain`, predictions tables unless
`--rescore`, and the explainers resume image by image, so the same line picks up where a
stopped run left off. Models run in the order given. Each range writes its own log,
`results/logs/<stage>_<model>_<arch>_<start>.log`.

On the Mac, `run_local.sh` wraps the same run with two checks first:

```bash
mkdir -p results/logs
nohup caffeinate -is bash run_local.sh >> results/logs/run_local.log 2>&1 &
tail -f results/logs/run_local.log
```

It checks the stack and reproduces the 24 stored records. Then it runs ResNet-50 and the
baseline, writes their summaries and figures, and only then starts the other four depths.
`bash run_local.sh resnet101` runs one depth. `BATCH=128 PARALLEL=4` in front suits a
machine with a GPU.

Stage by stage:

```bash
python code/run_all.py download
python code/run_all.py train   --archs resnet50        # skipped if the checkpoint exists
python code/run_all.py score   --archs resnet50
python code/run_all.py explain --archs resnet50 --batch 128 --parallel 4 --stability 200
python code/run_all.py report  --archs resnet50
```

Or one stage at a time:

```bash
python code/lime_run.py --arch resnet50 --first 0 --last 900 --batch 128
python code/shap_run.py --arch resnet50 --status
python code/lime_run.py --check          # reproduce the 24 stored records
```

Figures, once there are rows to draw from:

```bash
python code/figures_overlap.py  --arch resnet50                 # best and worst by overlap
python code/figures_overlap.py  --arch resnet50 --rank-by r2    # by lime's surrogate fit
python code/figures_accuracy.py --arch resnet50                 # surest and least sure
python code/figures_accuracy.py --arch resnet50 --only wrong    # confidently wrong cases
python code/figures_overlap.py  --arch resnet50 --summary       # whole-split panels as well
python code/figures_accuracy.py --family                        # accuracy and overlap by depth
```

Both read the stored rows, including the per-region weights, so no image is explained
again. Every figure is one row per image: the photo, then one map per method. Colour
always means the method. Accuracy and overlap are never put on two scales in one plot;
they are two panels sharing one axis.

Useful facts:

- Any run can be stopped and started again. It skips what is already on disk and loses at
  most the image in flight.
- A `--first/--last` range writes its own file, so ranges can run side by side on one
  machine or across machines. Put the CSVs in one folder before `report.py`.
- The image order is a fixed shuffle, so a partial run is still a random sample.
- The baseline is the stock ImageNet ResNet-50. It is explained over the test split like
  the fine-tuned models. `--split both` adds the 3,680 trainval images, which it has never
  seen either.
- A range that fails ten images in a row stops, so a fault such as the GPU running out of
  memory shows up as a failed range instead of a quietly empty one.

## Windows

- Install Python 3.12, then torch with a CUDA build. `code/requirements.txt` says which
  one from what `nvidia-smi` prints. Check `torch.cuda.is_available()` is `True`.
- `pip install -r code/requirements.txt` for the rest.
- Nothing here needs bash. `run_all.py` starts the parallel ranges itself.
- If the training loaders hang, pass `--workers 0`.
- Turn off sleep while a long run is going. There is no `caffeinate` on Windows.
- Follow one range with `Get-Content results/logs/lime_run_finetuned_resnet50_0.log -Tail 3 -Wait`.
- If a log shows CUDA out of memory, start again with `--batch 64`.
- Paths are worked out from the file locations, so the folder can be renamed or moved.
  The dataset is the exception: download it again, or pass `--data-root`.
- On the Mac, `data/oxford-iiit-pet` is a link to the copy in `../lime_method/data`, so
  nothing was downloaded twice. A link like that does not survive a copy to Windows.

## Moving this folder to another machine

Clone the repository, then copy the five checkpoints into `results/checkpoints/`. They are
not in git, being about 630 MB. On the new machine:

```bash
python code/run_all.py download
python code/lime_run.py --check
```

`check` re-explains the 24 stored records and compares them with the saved weights. It is
the quickest proof that the new machine explains like the old one. A different GPU or
torch version changes the last digits of the weights, so the exact count can fall short
there. The second count, same superpixels, class and top five, should still be 24/24.

## Results already here

- `results/checkpoints/` fine-tuned weights, one per depth.
- `results/scores/finetuned_resnet50.csv` and `results/scores/baseline.csv` are earlier
  scoring runs, kept so the explanations can start without rescoring.
- `results/lime/records/` the 24 stored LIME explanations from the grouped study, used by
  `lime_run.py --check`.
- Older runs live in `../lime_method/outputs` and are untouched. `report.py` can read the
  LIME CSVs from there if they are copied into `results/lime/`; they carry no weights or
  curves, so they only contribute overlap.
