$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$repoRoot = (Resolve-Path (Join-Path (Join-Path $here "..") "..")).Path
$py = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $py)) {
    throw "Venv not found at $py. Create .venv at repo root or edit run_local.ps1."
}
Set-Location -LiteralPath $here
& $py (Join-Path $here "app.py")
