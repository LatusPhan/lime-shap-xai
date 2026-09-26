# Full local run of the LIME and SHAP pipeline on the GPU, pushing results to GitHub
# after each model finishes. Every stage resumes, so running it again continues.
#
#   .\run_and_push.ps1
#   .\run_and_push.ps1 -Archs resnet50 -Parallel 2
#   .\run_and_push.ps1 -NoBaseline                the fine-tuned depths only
param(
    [string[]]$Archs = @("resnet18", "resnet34", "resnet50", "resnet101", "resnet152"),
    [int]$Parallel = 4,
    [int]$Batch = 128,
    [int]$Stability = 200,
    [int]$Workers = 4,
    [switch]$NoBaseline
)
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"

# Clear a stale git lock left by an interrupted git command
if (Test-Path .git\index.lock) { Remove-Item .git\index.lock -Force }

# Sync code and packages
Write-Host "=== $(Get-Date -Format HH:mm) syncing with GitHub"
git pull --rebase --autostash
if ($LASTEXITCODE) { exit 1 }
& $py -m pip install -q -r code\requirements.txt
if ($LASTEXITCODE) { exit 1 }

# Dataset (skipped if already extracted)
Write-Host "=== $(Get-Date -Format HH:mm) dataset"
& $py code\run_all.py download
if ($LASTEXITCODE) { exit 1 }

# Train only the depths with no checkpoint on this machine
$untrained = $Archs | Where-Object { -not (Test-Path "results\checkpoints\$($_)_pets.pt") }
if ($untrained) {
    Write-Host "=== $(Get-Date -Format HH:mm) training $($untrained -join ', ')"
    & $py code\run_all.py train --archs @untrained --workers $Workers
    if ($LASTEXITCODE) { exit 1 }
}

# Score only the depths with no score table
$unscored = $Archs | Where-Object { -not (Test-Path "results\scores\finetuned_$($_).csv") }
if ($unscored) {
    Write-Host "=== $(Get-Date -Format HH:mm) scoring $($unscored -join ', ')"
    & $py code\run_all.py score --archs @unscored --batch $Batch
    if ($LASTEXITCODE) { exit 1 }
}

# Explain, summarise, draw and push, one depth at a time
foreach ($arch in $Archs) {
    Write-Host "=== $(Get-Date -Format HH:mm) LIME and SHAP on $arch, $Parallel ranges at a time"
    & $py code\run_all.py explain --archs $arch --batch $Batch --parallel $Parallel --stability $Stability
    if ($LASTEXITCODE) { Write-Host "Explain failed on $arch. See results\logs"; exit 1 }

    & $py code\report.py --models finetuned --archs $arch
    & $py code\figures_overlap.py --arch $arch --summary
    & $py code\figures_accuracy.py --arch $arch

    git add results
    git commit -m "Results for $arch $(Get-Date -Format yyyy-MM-dd)"
    git pull --rebase --autostash
    git push
}

# The stock ImageNet ResNet-50, over the same test images as the fine-tuned depths
if (-not $NoBaseline) {
    Write-Host "=== $(Get-Date -Format HH:mm) LIME and SHAP on the baseline, $Parallel ranges at a time"
    & $py code\run_all.py explain --models baseline --batch $Batch --parallel $Parallel --stability $Stability
    if ($LASTEXITCODE) { Write-Host "Explain failed on the baseline. See results\logs"; exit 1 }

    & $py code\report.py --models baseline
    & $py code\figures_overlap.py --model baseline --arch resnet50 --summary
    & $py code\figures_accuracy.py --model baseline --arch resnet50

    git add results
    git commit -m "Results for the baseline $(Get-Date -Format yyyy-MM-dd)"
    git pull --rebase --autostash
    git push
}

# Summary across every model, and the family figure across the depths
Write-Host "=== $(Get-Date -Format HH:mm) family summary"
$models = if ($NoBaseline) { @("finetuned") } else { @("finetuned", "baseline") }
& $py code\report.py --models @models --archs @Archs
& $py code\figures_accuracy.py --family --archs @Archs
git add results
git commit -m "Family summary $(Get-Date -Format yyyy-MM-dd)"
git pull --rebase --autostash
git push
Write-Host "=== $(Get-Date -Format HH:mm) done: results\summary and results\figures are on GitHub"
