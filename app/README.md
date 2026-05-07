# Gradio Demo (Hugging Face Spaces)

## Local run

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -e .[demo]
.venv\Scripts\python app/gradio_app.py
```

The checked-in local app looks for `artifacts/model.joblib` first, then `artifacts/model.keras`.
Use the sklearn placeholder model for lightweight portfolio-safe validation, or provide a TensorFlow `.keras`
artifact for real-photo model testing.
If you plan to run the TensorFlow `.keras` path, install ML extras as well: `pip install -e .[ml,demo]`.

For Keras models trained at a non-default size, keep `artifacts/run_config.json` next to the model (written by `train`) or set `SNAKE_DETECTOR_IMAGE_SIZE` when launching the demo.

## Hugging Face Spaces setup

Spaces need a module with **`demo`** at import time and no **`launch()`** at import. This repo supports:

- Root [`app.py`](../app.py) - thin re-export of `demo` from `app/gradio_app.py` (works with HF's default `app.py`).
- [`app/gradio_app.py`](gradio_app.py) - set `app_file: app/gradio_app.py` in the Space README if you want the entry under `app/` only.

Bootstrap: [`hf_bootstrap.py`](hf_bootstrap.py) downloads the older placeholder-safe [release `model.joblib`](https://github.com/mmaitland300/Snake-detector/releases/download/v1.0.0/model.joblib) when missing (idempotent, timeout, atomic write). See [docs/hf_space.md](../docs/hf_space.md).

The current public Hugging Face Space uses a Space-side Keras model (`model.keras`) and deployment config documented in [docs/releases/v1.1.0-real-dataset.md](../docs/releases/v1.1.0-real-dataset.md). The Keras artifact is large enough that it belongs in Space files and GitHub Releases, not normal git history.

[`requirements.txt`](../requirements.txt) installs `.[demo]` from the Space root.

After deploy, set portfolio **`NEXT_PUBLIC_SNAKE_DEMO_URL`** to the **direct app URL** `https://mmaitland-snake-detector-demo.hf.space` (not the `huggingface.co/spaces/...` page). Details: [docs/hf_space.md](../docs/hf_space.md).

**Manual artifact fallback:** if the placeholder download fails, add `artifacts/model.joblib` via the Space Files tab. For the current real-photo TensorFlow demo path, keep `model.keras` and `deployment_config.json` in Space/release storage and install `.[ml,demo]` instead (heavier than CPU sklearn-only Spaces).

## Cold-start note

Free tier may sleep. Keep both:
- embedded iframe for in-page demo
- direct external "Open Demo" link as fallback
