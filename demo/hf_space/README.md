# Snake vs non-snake (demo)

**Disclaimer:** For demonstration only. Not a safety device. Do not use for medical, legal, or wildlife handling decisions.

## What the score means

This Space runs a small neural network that outputs two numbers that sum to 100%: how much the model leans toward **snake** versus **not a snake** for the *whole photo*. The plain-English line (**Likely a snake** / **Likely not a snake**) applies a fixed cutoff: if snake probability is at least the configured threshold (for example 76%), the demo labels it snake. That cutoff was chosen on validation data for this project; it is not universal truth. **Wide or zoomed-out pictures** are harder: the image is resized to a small square before scoring, so a tiny snake in the frame can be missed. Bounding-box detection is not part of this demo.

## Hugging Face Space setup

1. Create a **Gradio** Space (same files as this folder).
2. Upload **`app.py`**, **`requirements.txt`**, and your trained **`model.keras`**.
3. Use a **`deployment_config.json`** whose **`model_path`** is **`model.keras`** (same directory as `app.py`). The copy in this repo uses a path relative to the monorepo for local runs; on the Space it must point at the uploaded weights file.
4. Optional: set environment variables instead of or in addition to JSON: `SNAKE_DETECTOR_THRESHOLD`, `SNAKE_DETECTOR_IMAGE_SIZE`, `SNAKE_DETECTOR_MODEL_PATH`.

## Local run (from monorepo)

See the module docstring in `app.py` or run `.\run_local.ps1` from this directory with the repo `.venv` at the root.
