$ErrorActionPreference = "Stop"

Set-Location "C:\dev\Cursor Projects\Snake-detector"

function Invoke-DownloadManifestJson {
    $stdout = & .\.venv\Scripts\python -m snake_detector.cli download-manifest --manifest-path $manifest --output-dir $rawDir 2>&1
    $text = ($stdout | ForEach-Object { $_.ToString() }) -join "`n"
    Write-Host $text
    $summary = $null
    try {
        $summary = $text | ConvertFrom-Json
    } catch {
        # fall through; handle below
    }
    if ($null -ne $summary) {
        $line = "  -> downloaded=$($summary.downloaded) skipped_existing=$($summary.skipped_existing) failed=$($summary.failed) output_dir=$($summary.output_dir)"
        if ($LASTEXITCODE -ne 0) {
            Write-Host $line -ForegroundColor Yellow
        } else {
            Write-Host $line -ForegroundColor DarkCyan
        }
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n========== download-manifest failed (exit $LASTEXITCODE); see JSON above ==========" -ForegroundColor Yellow
        throw "download-manifest exited with code ${LASTEXITCODE} (nonzero failed count or other error). Re-run to resume skipped/failed rows."
    }
}

$manifest = "data/manifests/collection_main_v1.csv"
$rawDir = "data/raw_collection_main_v1"
$preSnapshot = "data/manifests/snapshots/collection_main_v1_pre_download.csv"

if (-not (Test-Path $manifest)) {
    throw "Manifest not found: $manifest. Run collection_main_v1_01_collect.ps1 first."
}

New-Item -ItemType Directory -Force "data/manifests/snapshots" | Out-Null
if (-not (Test-Path -LiteralPath $preSnapshot)) {
    Copy-Item -LiteralPath $manifest -Destination $preSnapshot
    Write-Host "Saved pre-download manifest snapshot: $preSnapshot" -ForegroundColor DarkCyan
} else {
    Write-Host "Pre-download snapshot already exists; not overwriting: $preSnapshot" -ForegroundColor DarkCyan
}

Invoke-DownloadManifestJson

Write-Host ""
Write-Host "Phase 2 complete." -ForegroundColor Green
Write-Host "Download finished. If needed, you can safely rerun this script to resume skipped/failed downloads."
Write-Host "When ready, run collection_main_v1_03_split_train_eval.ps1"
