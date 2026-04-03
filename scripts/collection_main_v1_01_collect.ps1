$ErrorActionPreference = "Stop"

Set-Location "C:\dev\Cursor Projects\Snake-detector"

function Run-Python {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$PythonArgs
    )
    & .\.venv\Scripts\python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        # -join drops quotes; long --user-agent may look split here even when Python received one argv.
        throw "Python command failed with exit code ${LASTEXITCODE}. Re-run the failing line from this script (see --user-agent quoting)."
    }
}

function Invoke-CollectInatJson {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$PythonArgs
    )
    $stdout = & .\.venv\Scripts\python @PythonArgs 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: collect-inat (see --user-agent quoting)."
    }
    $lines = @($stdout)
    for ($li = $lines.Count - 1; $li -ge 0; $li--) {
        $t = $lines[$li].ToString().Trim()
        if ($t.StartsWith("{") -and $t.EndsWith("}")) {
            try {
                return $t | ConvertFrom-Json
            } catch {
                continue
            }
        }
    }
    throw "Could not parse JSON summary from collect-inat output."
}

$manifest = "data/manifests/collection_main_v1.csv"
$artifactDir = "artifacts/collection_main_v1"
# Replace with your email so API operators can identify traffic (helps with some 403/WAF cases).
# Single-quoted string; pass as "--user-agent `"$InatUserAgent`"" on each line so spaces/parens stay one CLI argument.
$InatUserAgent = 'snake-detector/0.1 (+https://github.com/mmaitland300/Snake-detector; mailto:mmaitland300@gmail.com)'

# Default: checkpointed Serpentes pulls (--flush-every-page). Set $false for legacy single collect-inat (fragile on 403).
$UseSnakeChunks = $true
$SnakeTargetRows = 4500
$SnakeChunkStartPage = 1
$SnakePagesPerChunk = 20
$SnakePerPage = 30
# Safety cap on chunk iterations (avoids infinite loops if counts never reach target).
$SnakeMaxChunks = 200

if (Test-Path $manifest) {
    if (-not $UseSnakeChunks) {
        throw "Manifest already exists: $manifest. Use a fresh filename, remove it, or set `$UseSnakeChunks = `$true to resume chunked snake collection."
    }
}

New-Item -ItemType Directory -Force "data/manifests/snapshots" | Out-Null
New-Item -ItemType Directory -Force $artifactDir | Out-Null

# 1. Collect snake positives
if ($UseSnakeChunks) {
    $startPage = $SnakeChunkStartPage
    for ($iter = 0; $iter -lt $SnakeMaxChunks; $iter++) {
        $snakeCount = 0
        if (Test-Path $manifest) {
            $snakeCount = @(
                Import-Csv -LiteralPath $manifest -Encoding utf8 | Where-Object { $_.label -eq "snake" }
            ).Count
        }
        if ($snakeCount -ge $SnakeTargetRows) {
            Write-Host "Snake row target reached ($snakeCount >= $SnakeTargetRows). Stopping snake chunk loop." -ForegroundColor Green
            break
        }
        $remaining = $SnakeTargetRows - $snakeCount
        if ($remaining -le 0) {
            break
        }

        $append = Test-Path $manifest
        $chunkArgs = @(
            "-m", "snake_detector.cli", "collect-inat",
            "--label", "snake",
            "--taxon-name", "Serpentes",
            "--manifest-path", $manifest,
            "--max-images", "$remaining",
            "--per-page", "$SnakePerPage",
            "--start-page", "$startPage",
            "--max-pages", "$SnakePagesPerChunk",
            "--flush-every-page",
            "--user-agent", $InatUserAgent
        )
        if ($append) {
            $chunkArgs += "--append"
        }

        Write-Host "Snake chunk: start_page=$startPage max_pages=$SnakePagesPerChunk snake_rows_so_far=$snakeCount remaining_cap=$remaining" -ForegroundColor Cyan
        $summary = Invoke-CollectInatJson @chunkArgs

        if ($summary.pages_fetched -eq 0) {
            Write-Host "No observation pages returned this chunk (likely end of API results). Stopping snake loop." -ForegroundColor Yellow
            break
        }

        $startPage = [int]$summary.next_page
    }
    if (Test-Path $manifest) {
        $finalSnake = @(
            Import-Csv -LiteralPath $manifest -Encoding utf8 | Where-Object { $_.label -eq "snake" }
        ).Count
        if ($finalSnake -lt $SnakeTargetRows) {
            Write-Host "Snake rows in manifest: $finalSnake (target $SnakeTargetRows). Raise `$SnakeMaxChunks, adjust filters, or resume with a higher `$SnakeChunkStartPage / existing manifest." -ForegroundColor Yellow
        }
    }
} else {
    Run-Python -m snake_detector.cli collect-inat --label snake --taxon-name Serpentes --manifest-path $manifest --max-images $SnakeTargetRows --per-page $SnakePerPage --user-agent "$InatUserAgent"
}

# 2. Collect no_snake negatives
Run-Python -m snake_detector.cli collect-inat --label no_snake --taxon-name Mammalia --manifest-path $manifest --max-images 1000 --append --user-agent "$InatUserAgent"
Run-Python -m snake_detector.cli collect-inat --label no_snake --taxon-name Aves --manifest-path $manifest --max-images 1000 --append --user-agent "$InatUserAgent"
Run-Python -m snake_detector.cli collect-inat --label no_snake --taxon-name Anura --manifest-path $manifest --max-images 800 --append --user-agent "$InatUserAgent"
Run-Python -m snake_detector.cli collect-inat --label no_snake --taxon-name Actinopterygii --manifest-path $manifest --max-images 800 --append --user-agent "$InatUserAgent"
Run-Python -m snake_detector.cli collect-inat --label no_snake --taxon-name Testudines --manifest-path $manifest --max-images 450 --append --user-agent "$InatUserAgent"
Run-Python -m snake_detector.cli collect-inat --label no_snake --taxon-name Caudata --manifest-path $manifest --max-images 300 --append --user-agent "$InatUserAgent"
Run-Python -m snake_detector.cli collect-inat --label no_snake --taxon-name Crocodylia --manifest-path $manifest --max-images 150 --append --user-agent "$InatUserAgent"
Run-Python -m snake_detector.cli collect-inat --label no_snake --taxon-name Lepidoptera --manifest-path $manifest --max-images 250 --append --user-agent "$InatUserAgent"
Run-Python -m snake_detector.cli collect-inat --label no_snake --taxon-name Odonata --manifest-path $manifest --max-images 100 --append --user-agent "$InatUserAgent"
Run-Python -m snake_detector.cli collect-inat --label no_snake --taxon-name Araneae --manifest-path $manifest --max-images 200 --append --user-agent "$InatUserAgent"

Write-Host ""
Write-Host "Manifest sanity checks" -ForegroundColor Cyan

Import-Csv -LiteralPath $manifest -Encoding utf8 |
    Group-Object label |
    Select-Object Name, Count |
    Format-Table -AutoSize

Import-Csv -LiteralPath $manifest -Encoding utf8 |
    Measure-Object |
    Select-Object Count

Import-Csv -LiteralPath $manifest -Encoding utf8 |
    Where-Object { -not $_.observation_id -or -not $_.image_id -or -not $_.image_url } |
    Select-Object -First 10 |
    Format-Table -AutoSize

Import-Csv -LiteralPath $manifest -Encoding utf8 |
    Group-Object label, provider, image_id |
    Where-Object { $_.Count -gt 1 } |
    Select-Object -First 10 Count, Name |
    Format-Table -AutoSize

Write-Host ""
Write-Host "Phase 1 complete." -ForegroundColor Green
Write-Host "Review the manifest checks before running collection_main_v1_02_download.ps1"
