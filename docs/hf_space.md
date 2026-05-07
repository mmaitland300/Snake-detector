# Hugging Face Space (Gradio, CPU)

This repo is **Gradio Spaces-compatible**: the Gradio runtime needs a module that defines **`demo`** at import time and does **not** call `launch()` at module scope.

## Current Live Space

- Space repo: `https://huggingface.co/spaces/mmaitland/snake-detector-demo`
- Direct app URL: `https://mmaitland-snake-detector-demo.hf.space`
- Current live model: Space-side `model.keras`
- Effective config: `image_size=160`, `threshold=0.76`, `preprocessing=inception_v3_preprocess_input`
- Matching GitHub release package: [v1.1.0-real-dataset](releases/v1.1.0-real-dataset.md)

The live Space is a bounded binary `snake` vs `no_snake` demo trained on iNaturalist-sourced real photos. It is not species identification, field-ready wildlife guidance, or a safety device.

## Entrypoints (pick one)

- **Default (zero extra config):** root [`app.py`](../app.py) re-exports `demo` from [`app/gradio_app.py`](../app/gradio_app.py). Hugging Face's Gradio template often looks for `app.py` first.
- **Repo layout preference:** set **`app_file: app/gradio_app.py`** in the Space README metadata (see below). That file loads [`app/hf_bootstrap.py`](../app/hf_bootstrap.py) for the older placeholder bootstrap, then sets `demo = build_demo()` at module level.

Either way: expose **`demo = build_demo()`** for Spaces and avoid calling **`launch()`** during the import that Spaces performs. This repo uses `demo.launch()` only under `if __name__ == "__main__"` for local runs.

## Model Artifacts and Bootstrap

There are now two model lanes:

- **Current live Space:** `model.keras` is stored in the Hugging Face Space files and documented in [v1.1.0-real-dataset](releases/v1.1.0-real-dataset.md).
- **Checked-in repository bootstrap:** [`app/hf_bootstrap.py`](../app/hf_bootstrap.py) still downloads the older placeholder-safe [`v1.0.0 model.joblib`](https://github.com/mmaitland300/Snake-detector/releases/download/v1.0.0/model.joblib) when `artifacts/model.joblib` is missing.

That split is intentional for now: the real Keras model is large, belongs in release/Space storage, and needs TensorFlow dependencies; the lightweight repo bootstrap remains useful for local smoke tests and placeholder-safe engineering proof.

### Placeholder bootstrap behavior

Free CPU Spaces **sleep and restart**; startup logic is designed to be safe across restarts:

- **Idempotent:** if `artifacts/model.joblib` exists and is at least 512 bytes, download is skipped; smaller files are removed and replaced.
- **Timeout:** HTTP download uses a bounded timeout of 120 seconds.
- **Atomic install:** bytes are written to a temp file, then renamed into place.
- **Clear failures:** network, HTTP, or suspiciously small payloads raise `RuntimeError` with a short message pointing at the Space **Files** tab fallback.

If the GitHub repository runtime is later migrated fully to the real Keras release path, update `requirements.txt`, `app/hf_bootstrap.py`, `app/gradio_app.py`, and this document together.

## Create the Space

1. Sign in at [Hugging Face](https://huggingface.co) and open [New Space](https://huggingface.co/new-space).
2. Choose **Gradio** as the SDK and **CPU basic** hardware unless the TensorFlow runtime needs more.
3. Connect your code through the Space Git repository or the upload/sync flow available for your account.
4. Ensure the Space installs the dependencies required by the selected runtime.
5. Set the app file if you are not using the default `app.py`: in the Space README front matter, set `app_file` to `app/gradio_app.py`.
6. If the Space tracks a Git branch or mirrors from GitHub, point it at this repository's default branch (`main`).

## Public URL for the Portfolio

Use the **running app URL**, not the Space repository page.

**Preferred for CTAs and `NEXT_PUBLIC_SNAKE_DEMO_URL`:**

`https://mmaitland-snake-detector-demo.hf.space`

**Less ideal for a "Try live demo" button** because it opens the repo/gallery page:

`https://huggingface.co/spaces/mmaitland/snake-detector-demo`

In [mmaitland-portfolio](https://github.com/mmaitland300/mmaitland-portfolio), set:

`NEXT_PUBLIC_SNAKE_DEMO_URL=https://mmaitland-snake-detector-demo.hf.space`

That enables "Try live demo" and sends visitors directly to the app.

### Portfolio copy note ("shipped" vs live demo)

If the site shows **shipped** when no demo URL is set, treat that as **case study / repo shipped**, not "live app is up," unless the surrounding copy makes that explicit. After `NEXT_PUBLIC_SNAKE_DEMO_URL` is set, the live CTA should use the **hf.space** URL above.

## Optional README YAML

```yaml
---
title: Snake Detector Demo
emoji: snake
colorFrom: gray
colorTo: green
sdk: gradio
app_file: app/gradio_app.py
pinned: false
---
```

Use `app_file: app.py` instead if you prefer the default root entry. Adjust `sdk_version` only if the Space template requires a pinned Gradio version.

## Manual Model Fallback

For the current real-photo Space, keep `model.keras` and `deployment_config.json` in the Space **Files** area beside the app. Do not commit the large Keras model to git.

For the older placeholder bootstrap path, if the release download is blocked, add `artifacts/model.joblib` via the Space **Files** tab.

## Local Check

```bash
pip install -e ".[demo]"
python app.py
# or:
python app/gradio_app.py
```

Hugging Face imports the configured app file and uses `demo` only; `launch()` runs only when you execute the file as `__main__`.

## Handoff Checklist

1. Make this repo Spaces-compatible (done: `demo` + bootstrap + [`requirements.txt`](../requirements.txt)).
2. Keep the Space-side real model files (`model.keras`, `deployment_config.json`) aligned with the GitHub Release package.
3. Open `https://mmaitland-snake-detector-demo.hf.space` and sanity-check the UI.
4. Set `NEXT_PUBLIC_SNAKE_DEMO_URL` to that `hf.space` URL in Vercel and `.env.local` for the portfolio; redeploy the portfolio.

**One-line reminder:** Prefer `app_file: app/gradio_app.py` if you want all Space logic under `app/`; otherwise keep the default root `app.py`. Either way, expose **`demo = build_demo()`** at module scope and do not call **`launch()`** on import.
