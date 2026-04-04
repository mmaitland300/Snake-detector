# Snake vs non-snake (demo)

**Disclaimer:** For demonstration only. Not a safety device. Do not use for medical, legal, or wildlife handling decisions.

## What the score means

This Space runs a small neural network that outputs a **snake likelihood** for the *whole photo* (one main percentage in the UI). The headline (**Likely a snake** / **Likely not a snake** / **Undetermined**) uses **two** cutoffs: confident snake when likelihood is high, confident not-snake when low, and **Undetermined** in between so the demo does not pretend the model is sure when it is not. Thresholds are configurable (defaults 76% / 24% for high / low bands). **Wide or zoomed-out pictures** are harder: the image is resized to a small square before scoring, so a tiny snake in the frame can be missed. Bounding-box detection is not part of this demo. The project **CLI** still uses a single threshold for binary labels; this demo's three-way wording is for visitors only.

## Hugging Face Space setup

A Space is **its own Git repository** on Hugging Face. The Hub's "clone -> add `app.py` -> push" flow is correct. Keep this monorepo folder as the **source of truth** for code; use a **separate clone directory** next to the project (not inside `Snake-detector/`) so you do not nest two `.git` roots.

### Step-by-step (recommended)

1. On [huggingface.co/new-space](https://huggingface.co/new-space), create a **Gradio** Space and note the repo URL, e.g. `https://huggingface.co/spaces/<you>/<space-name>`.
2. **Clone** that repo to a **sibling** of this monorepo, e.g.  
   `C:\dev\Cursor Projects\snake-detector-demo`  
   (same parent as `Snake-detector`, **not** inside `Snake-detector\demo\hf_space`).
   ```powershell
   cd "C:\dev\Cursor Projects"
   git clone https://huggingface.co/spaces/<you>/<space-name>.git
   ```
3. **Copy files into the clone** (manual copy is fine), or from `demo\hf_space` run:
   ```powershell
   .\publish_to_hf_space.ps1 -SpaceRepoPath "C:\dev\Cursor Projects\<space-folder>"
   ```
   That script copies **`app.py`**, **`requirements.txt`**, **`deployment_config.json`** (from committed **`deployment_config.hf.json`**, which sets `"model_path": "model.keras"`), and **`model.keras`** if your local `deployment_config.json` path resolves to an existing file under `artifacts/`.
4. **`README.md` in the clone:** Hugging Face created a YAML header (`sdk: gradio`, etc.). **Do not replace the whole file** with only this folder's README, or the Space may lose SDK metadata. Keep the `---` / `sdk:` block at the top; add the disclaimer and any extra sections below it.
5. **Large `model.keras` (run inside the Space clone before the first commit that adds weights):**
   ```powershell
   git lfs install   # once per machine, if needed
   cd "C:\dev\Cursor Projects\snake-detector-demo"   # your clone path
   git lfs track "*.keras"
   ```
   Confirm `.gitattributes` contains a `*.keras` LFS line, then commit `.gitattributes` with the model (either order is fine as long as LFS is active before the large file is added). Alternatively upload `model.keras` via the Hub **Files** tab.
6. **Optional env** on the Space: `SNAKE_DETECTOR_THRESHOLD` (high band for tri-state UX), `SNAKE_DETECTOR_LOW_THRESHOLD`, `SNAKE_DETECTOR_IMAGE_SIZE`, `SNAKE_DETECTOR_MODEL_PATH` (see `app.py`). Config file keys: `threshold`, optional `low_threshold` (defaults to `1 - threshold` if omitted).
7. Commit and **`git push`** from the clone; wait for the build on the **App** tab.

`hf download <space> --repo-type=space` only caches Hub files; it does **not** replace this clone-and-push workflow for **your** edits and weights.

### Files in this folder vs on the Space

| File | In monorepo | On Hugging Face Space |
|------|-------------|------------------------|
| `app.py` | Yes | Yes (copy from here) |
| `requirements.txt` | Yes | Yes |
| `deployment_config.json` | Points at `../../artifacts/...` for local runs | Must be **`model.keras`** next to `app.py` - use **`deployment_config.hf.json`** as the template (sync script writes `deployment_config.json` in the clone) |
| `deployment_config.hf.json` | Yes (Space template only) | Not uploaded; becomes `deployment_config.json` in the clone |
| `model.keras` | Usually ignored / under `artifacts/` locally | Required in Space root (LFS if large) |

### Shorter checklist (upload-only)

1. Create a **Gradio** Space.
2. Upload **`app.py`**, **`requirements.txt`**, **`model.keras`**, and **`deployment_config.json`** with **`model_path`** set to **`model.keras`** (same as `deployment_config.hf.json` here).
3. Merge **README** YAML + disclaimer as above.
4. Optional: environment variables `SNAKE_DETECTOR_THRESHOLD`, `SNAKE_DETECTOR_LOW_THRESHOLD`, `SNAKE_DETECTOR_IMAGE_SIZE`, `SNAKE_DETECTOR_MODEL_PATH`.

## Local run (from monorepo)

See the module docstring in `app.py` or run `.\run_local.ps1` from this directory with the repo `.venv` at the root.
