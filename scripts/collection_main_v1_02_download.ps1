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
$preSnapshot = "data/manifests/snapshots/collection_main_v1_pre_download.csv"

if (-not (Test-Path $manifest)) {
    throw "Manifest not found: $manifest. Run collection_main_v1_01_collect.ps1 first."
}

New-Item -ItemType Directory -Force "data/manifests/snapshots" | Out-Null
Copy-Item -LiteralPath $manifest -Destination $preSnapshot -Force

Run-Python -m snake_detector.cli download-manifest --manifest-path $manifest --output-dir $rawDir

Write-Host ""
Write-Host "Phase 2 complete." -ForegroundColor Green
Write-Host "Download finished. If needed, you can safely rerun this script to resume skipped/failed downloads."
Write-Host "When ready, run collection_main_v1_03_split_train_eval.ps1"
