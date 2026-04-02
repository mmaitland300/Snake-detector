# Gradio Demo (Hugging Face Spaces)

## Local run

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -e .[demo]
.venv\Scripts\python app/gradio_app.py
```

The local app looks for `artifacts/model.joblib` first, then `artifacts/model.keras`.
Use the sklearn placeholder model for portfolio-safe validation, or provide a TensorFlow `.keras`
artifact if you later publish a rights-cleared model.
If you plan to run the TensorFlow `.keras` path, install ML extras as well: `pip install -e .[ml,demo]`.

For Keras models trained at a non-default size, keep `artifacts/run_config.json` next to the model (written by `train`) or set `SNAKE_DETECTOR_IMAGE_SIZE` when launching the demo.

## Hugging Face Spaces setup

Spaces need a module with **`demo`** at import time and no **`launch()`** at import. This repo supports:

- Root [`app.py`](../app.py) - thin re-export of `demo` from `app/gradio_app.py` (works with HF's default `app.py`).
- [`app/gradio_app.py`](gradio_app.py) - set `app_file: app/gradio_app.py` in the Space README if you want the entry under `app/` only.

Bootstrap: [`hf_bootstrap.py`](hf_bootstrap.py) downloads [release `model.joblib`](https://github.com/mmaitland300/Snake-detector/releases/download/v1.0.0/model.joblib) when missing (idempotent, timeout, atomic write). See [docs/hf_space.md](../docs/hf_space.md).

[`requirements.txt`](../requirements.txt) installs `.[demo]` from the Space root.

After deploy, set portfolio **`NEXT_PUBLIC_SNAKE_DEMO_URL`** to the **direct app URL** `https://<space-subdomain>.hf.space` (not the `huggingface.co/spaces/...` page). Details: [docs/hf_space.md](../docs/hf_space.md).

**Manual artifact fallback:** if download fails, add `artifacts/model.joblib` via the Space Files tab (do not commit large binaries to git). For a TensorFlow demo path, use `artifacts/model.keras` and install `.[ml,demo]` instead (heavier than CPU sklearn-only Spaces).

## Cold-start note

Free tier may sleep. Keep both:
- embedded iframe for in-page demo
- direct external "Open Demo" link as fallback
