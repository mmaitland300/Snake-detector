$ErrorActionPreference = "Stop"

Set-Location "C:\dev\Cursor Projects\Snake-detector"

function Run-Python {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$PythonArgs
    )
    & .\.venv\Scripts\python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}. Re-run the failing line from this script."
    }
}

$manifest = "data/manifests/collection_main_v1.csv"
$rawDir = "data/raw_collection_main_v1"
$splitDir = "data/split_collection_main_v1"
$artifactDir = "artifacts/collection_main_v1"

$trainMetrics = "$artifactDir/metrics_tf.json"
$trainConfusion = "$artifactDir/confusion_matrix.png"
$trainPanel = "$artifactDir/sample_predictions.png"
$trainPreds = "$artifactDir/sample_predictions.json"

$evalMetrics = "$artifactDir/metrics_eval_tf.json"
$evalConfusion = "$artifactDir/confusion_matrix_eval.png"
$evalPanel = "$artifactDir/sample_predictions_eval.png"
$evalPreds = "$artifactDir/sample_predictions_eval.json"

if (-not (Test-Path $manifest)) {
    throw "Manifest not found: $manifest. Run collection_main_v1_01_collect.ps1 first."
}
if (-not (Test-Path $rawDir)) {
    throw "Raw dataset directory not found: $rawDir. Run collection_main_v1_02_download.ps1 first."
}

New-Item -ItemType Directory -Force $artifactDir | Out-Null

Run-Python -m snake_detector.cli split --raw-dir $rawDir --split-dir $splitDir --manifest-path $manifest --group-by observation_id --train-split 0.8 --val-split 0.1 --seed 42

Run-Python -m snake_detector.cli train --split-dir $splitDir --model-path "$artifactDir/model.keras" --metrics-path $trainMetrics --confusion-matrix-path $trainConfusion --predictions-panel-path $trainPanel --predictions-manifest-path $trainPreds --image-size 160 --batch-size 16 --epochs 12 --learning-rate 0.0001 --seed 42 --backbone inceptionv3

Run-Python -m snake_detector.cli eval --split-dir $splitDir --model-path "$artifactDir/model.keras" --metrics-path $evalMetrics --confusion-matrix-path $evalConfusion --predictions-panel-path $evalPanel --predictions-manifest-path $evalPreds --image-size 160 --batch-size 16 --backbone inceptionv3

Write-Host ""
Write-Host "Phase 3 complete." -ForegroundColor Green
Write-Host "Artifacts written under $artifactDir"
