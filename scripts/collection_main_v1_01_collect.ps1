$ErrorActionPreference = "Stop"

Set-Location "C:\dev\Cursor Projects\Snake-detector"

function Run-Python {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$PythonArgs
    )
    $out = & .\.venv\Scripts\python @PythonArgs 2>&1
    $text = ($out | ForEach-Object { $_.ToString() }) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n========== Python failed (exit $LASTEXITCODE); full output ==========" -ForegroundColor Yellow
        Write-Host $text
        Write-Host "=======================================================================`n" -ForegroundColor Yellow
        throw "Python command failed with exit code ${LASTEXITCODE}. Output is printed above (see --user-agent quoting)."
    }
}

function Get-FirstJsonObject {
    param([string]$Text)
    # CLI prints json.dumps(..., indent=2). Match first balanced { ... } by depth, ignoring braces inside JSON strings.
    $start = $Text.IndexOf('{')
    if ($start -lt 0) {
        return $null
    }
    $depth = 0
    $inString = $false
    $escape = $false
    for ($i = $start; $i -lt $Text.Length; $i++) {
        $c = $Text[$i]
        if ($inString) {
            if ($escape) {
                $escape = $false
                continue
            }
            if ($c -eq [char]0x5C) {
                $escape = $true
                continue
            }
            if ($c -eq '"') {
                $inString = $false
            }
            continue
        }
        if ($c -eq '"') {
            $inString = $true
            continue
        }
        if ($c -eq '{') {
            $depth++
        } elseif ($c -eq '}') {
            $depth--
            if ($depth -eq 0) {
                return $Text.Substring($start, $i - $start + 1)
            }
        }
    }
    return $null
}

function Invoke-CollectInatJson {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$PythonArgs
    )
    $stdout = & .\.venv\Scripts\python @PythonArgs 2>&1
    $text = ($stdout | ForEach-Object { $_.ToString() }) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n========== collect-inat failed (exit $LASTEXITCODE); full output ==========" -ForegroundColor Yellow
        Write-Host $text
        Write-Host "========================================================================`n" -ForegroundColor Yellow
        # Script-scoped: Test-Path variable:artifactDir fails inside this function on Windows PowerShell 5.1.
        $logDir = $script:artifactDir
        if ($logDir) {
            New-Item -ItemType Directory -Force $logDir | Out-Null
            $errLog = Join-Path $logDir "last_collect_error.txt"
            $text | Set-Content -LiteralPath $errLog -Encoding utf8
            Write-Host "Also saved to $errLog" -ForegroundColor DarkYellow
        }
        throw "collect-inat exited with code ${LASTEXITCODE}. Full traceback is printed above."
    }
    $jsonText = Get-FirstJsonObject -Text $text
    if ($null -ne $jsonText) {
        try {
            return $jsonText | ConvertFrom-Json
        } catch {
            # fall through to error below
        }
    }
    $preview = if ($text.Length -gt 600) { $text.Substring(0, 600) + "..." } else { $text }
    throw "Could not parse JSON summary from collect-inat output. First output was:`n---`n$preview"
}

$manifest = "data/manifests/collection_main_v1.csv"
# Single integer: last collect-inat next_page after each successful chunk. Lets restarts skip manual $SnakeChunkStartPage edits.
$snakeCursorFile = [System.IO.Path]::ChangeExtension($manifest, ".snake_next_page.txt")
# If true and the cursor file exists, snake chunk 1 starts there; $SnakeChunkStartPage is ignored. Set false to force $SnakeChunkStartPage.
$SnakeResumeFromCursorFile = $true
$artifactDir = "artifacts/collection_main_v1"
# Replace with your email so API operators can identify traffic (helps with some 403/WAF cases).
# Single-quoted string; pass as "--user-agent `"$InatUserAgent`"" on each line so spaces/parens stay one CLI argument.
$InatUserAgent = 'snake-detector/0.1 (+https://github.com/mmaitland300/Snake-detector; mailto:mmaitland300@gmail.com)'

# Default: checkpointed Serpentes pulls (--flush-every-page). Set $false for legacy single collect-inat (fragile on 403).
$UseSnakeChunks = $true
# Set $true to skip the entire snake chunk loop and go straight to no_snake (e.g. pausing Serpentes after errors or enough positives).
$SkipSnakeChunkLoop = $true
$SnakeTargetRows = 4500
# Used when $SnakeResumeFromCursorFile is false or no .snake_next_page.txt exists yet. New manifest: use 1.
$SnakeChunkStartPage = 331
# Smaller chunks + smaller per_page reduce burst traffic and 403/WAF risk (vs 20 pages x 30 rows).
$SnakePagesPerChunk = 10
$SnakePerPage = 20
# Safety cap on chunk iterations (avoids infinite loops if counts never reach target).
$SnakeMaxChunks = 200
# Stop snake loop after this many chunks in a row with records_written=0 (duplicate-only API pages / no new rows).
# Enforced only when this value is > 0 (0 disables). Otherwise use a very large number.
$SnakeMaxConsecutiveZeroWriteChunks = 10

if (Test-Path $manifest) {
    if (-not $UseSnakeChunks) {
        throw "Manifest already exists: $manifest. Use a fresh filename, remove it, or set `$UseSnakeChunks = `$true to resume chunked snake collection."
    }
}

New-Item -ItemType Directory -Force "data/manifests/snapshots" | Out-Null
New-Item -ItemType Directory -Force $artifactDir | Out-Null
$snakeCursorParent = [System.IO.Path]::GetDirectoryName($snakeCursorFile)
if ($snakeCursorParent) {
    New-Item -ItemType Directory -Force -Path $snakeCursorParent | Out-Null
}

# 1. Collect snake positives
if ($SkipSnakeChunkLoop) {
    Write-Host "Skipping snake chunk loop (`$SkipSnakeChunkLoop = `$true). no_snake collection runs next." -ForegroundColor Cyan
} elseif ($UseSnakeChunks) {
    $startPage = $SnakeChunkStartPage
    if ($SnakeResumeFromCursorFile -and (Test-Path -LiteralPath $snakeCursorFile)) {
        $rawCursor = (Get-Content -LiteralPath $snakeCursorFile -Raw).Trim()
        $cursorPage = 0
        if ([int]::TryParse($rawCursor, [ref]$cursorPage) -and $cursorPage -ge 1) {
            $startPage = $cursorPage
            Write-Host "Snake pagination: starting at saved next_page=$startPage ($snakeCursorFile). Set `$SnakeResumeFromCursorFile = `$false to use `$SnakeChunkStartPage ($SnakeChunkStartPage)." -ForegroundColor Cyan
        }
    }
    $consecutiveZeroWriteChunks = 0
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

        if ([int]$summary.records_written -gt 0) {
            $consecutiveZeroWriteChunks = 0
        } else {
            $consecutiveZeroWriteChunks++
        }

        # CSV grows only when records_written > 0. zero_write_streak counts consecutive chunks with no new rows appended.
        Write-Host "  chunk_stats: records_written=$($summary.records_written) skipped_duplicate=$($summary.records_skipped_duplicate) collected_this_run=$($summary.records_collected) pages_fetched=$($summary.pages_fetched) next_page=$($summary.next_page) zero_write_streak=$consecutiveZeroWriteChunks" -ForegroundColor White

        # Manifest rows do not record API page numbers; persist next_page so the next process can resume without re-walking duplicate windows.
        Set-Content -LiteralPath $snakeCursorFile -Value ([string][int]$summary.next_page) -Encoding utf8

        if ($SnakeMaxConsecutiveZeroWriteChunks -gt 0 -and $consecutiveZeroWriteChunks -ge $SnakeMaxConsecutiveZeroWriteChunks) {
            Write-Host "Stopping snake loop: $SnakeMaxConsecutiveZeroWriteChunks consecutive chunks with records_written=0 (duplicate-only or filtered out). Next run will resume from $snakeCursorFile (next_page=$($summary.next_page)) unless you delete that file or set `$SnakeResumeFromCursorFile = `$false." -ForegroundColor Yellow
            break
        }

        $startPage = [int]$summary.next_page
        # Cool-down between Python chunk processes (back-to-back bursts can trigger 403/WAF).
        Start-Sleep -Seconds 5
    }
    if (Test-Path $manifest) {
        $finalSnake = @(
            Import-Csv -LiteralPath $manifest -Encoding utf8 | Where-Object { $_.label -eq "snake" }
        ).Count
        if ($finalSnake -lt $SnakeTargetRows) {
            Write-Host "Snake rows in manifest: $finalSnake (target $SnakeTargetRows). Options: raise `$SnakeMaxChunks / widen filters / resume with `$SnakeChunkStartPage from last JSON next_page / increase `$SnakeMaxConsecutiveZeroWriteChunks if API was only briefly duplicate-heavy." -ForegroundColor Yellow
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
