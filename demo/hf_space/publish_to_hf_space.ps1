<#
.SYNOPSIS
  Copy Space-ready files from this folder into a cloned Hugging Face Space repo.

.DESCRIPTION
  Use AFTER you `git clone https://huggingface.co/spaces/<user>/<space-name>` into a
  folder OUTSIDE this monorepo (sibling directory recommended).

.PARAMETER SpaceRepoPath
  Absolute or relative path to the root of the cloned Space repository (contains .git).

.EXAMPLE
  .\publish_to_hf_space.ps1 -SpaceRepoPath "C:\dev\Cursor Projects\snake-detector-demo"
#>
param(
    [Parameter(Mandatory = $true)]
    [string] $SpaceRepoPath
)

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot

if (-not (Test-Path -LiteralPath $SpaceRepoPath)) {
    throw "Space repo not found: $SpaceRepoPath. Clone the Space first (see README.md)."
}
if (-not (Test-Path -LiteralPath (Join-Path $SpaceRepoPath ".git"))) {
    throw "Not a git repo root (no .git): $SpaceRepoPath"
}

$dest = (Resolve-Path -LiteralPath $SpaceRepoPath).Path

$localCfg = Join-Path $here "deployment_config.json"
if (-not (Test-Path -LiteralPath $localCfg)) {
    throw "Missing $localCfg"
}
$cfg = Get-Content -LiteralPath $localCfg -Raw | ConvertFrom-Json
$modelRel = $cfg.model_path
$modelSrc = [System.IO.Path]::GetFullPath((Join-Path $here $modelRel))

Copy-Item -LiteralPath (Join-Path $here "app.py") -Destination (Join-Path $dest "app.py") -Force
Copy-Item -LiteralPath (Join-Path $here "requirements.txt") -Destination (Join-Path $dest "requirements.txt") -Force
Copy-Item -LiteralPath (Join-Path $here "deployment_config.hf.json") -Destination (Join-Path $dest "deployment_config.json") -Force

if (Test-Path -LiteralPath $modelSrc) {
    Copy-Item -LiteralPath $modelSrc -Destination (Join-Path $dest "model.keras") -Force
    Write-Host "Copied model.keras from monorepo artifacts."
}
else {
    Write-Warning "Model not found at:`n  $modelSrc`nCopy your trained model.keras into:`n  $dest`nbefore git add / push."
}

Write-Host @"

Copied app.py, requirements.txt, deployment_config.json (from deployment_config.hf.json).

Next steps (in the Space repo):
  1. README: keep the YAML frontmatter Hugging Face created (sdk: gradio). Append or merge the disclaimer from this folder's README.md; do not delete the --- sdk block.
  2. If model.keras is large: git lfs install && git lfs track "*.keras" (once per clone), then add/commit.
  3. git add app.py requirements.txt deployment_config.json README.md model.keras
  4. git commit -m "Deploy snake detector demo"
  5. git push

Then open the Space App tab and wait for the build.
"@
