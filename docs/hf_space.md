# Hugging Face Space (Gradio, CPU)

This repo is **Gradio Spaces-compatible**: the Gradio runtime needs a module that defines **`demo`** at import time and does **not** call `launch()` at module scope.

## Entrypoints (pick one)

- **Default (zero extra config):** root [`app.py`](../app.py) re-exports `demo` from [`app/gradio_app.py`](../app/gradio_app.py). Hugging Face's Gradio template often looks for `app.py` first.
- **Repo layout preference:** set **`app_file: app/gradio_app.py`** in the Space README metadata (see below). That file loads [`app/hf_bootstrap.py`](../app/hf_bootstrap.py) (artifact download), then sets `demo = build_demo()` at module level. Same behavior as using root `app.py`, without relying on a second file.

Either way: expose **`demo = build_demo()`** for Spaces and avoid calling **`launch()`** during the import that Spaces performs. (This repo uses `demo.launch()` only under `if __name__ == "__main__"` for local runs.)

## Model bootstrap (`app/hf_bootstrap.py`)

Free CPU Spaces **sleep and restart**; startup logic is designed to be safe across restarts:

- **Idempotent:** if `artifacts/model.joblib` exists and is at least 512 bytes, download is skipped; smaller files are removed and replaced (partial or bogus uploads).
- **Timeout:** HTTP download uses a bounded timeout (120s).
- **Atomic install:** bytes are written to a temp file, then renamed into place.
- **Clear failures:** network/HTTP/suspiciously small payloads raise `RuntimeError` with a short message pointing at the Space **Files** tab fallback.

Current artifact source: [GitHub Release v1.0.0 `model.joblib`](https://github.com/mmaitland300/Snake-detector/releases/download/v1.0.0/model.joblib). **Optional later improvement:** host the file on the Hugging Face Hub (e.g. repo or Space storage) to drop the GitHub dependency at cold start and align with the Space ecosystem. Not required for the first publish.

## Create the Space

1. Sign in at [Hugging Face](https://huggingface.co) and open [**New Space**](https://huggingface.co/new-space).
2. Choose **Gradio** as the SDK and **CPU basic** (free) hardware.
3. **Connect your code:** create the Space, then either:
   - work in the Space's **Git** repository (clone, push, build), and keep GitHub as your public source of truth by mirroring or copying changes when you cut a release, or
   - use whatever **sync or upload** flow the Space UI offers for your account (wording and options change; do not assume a specific one-click GitHub link until you see it in your UI).
4. Ensure the Space installs dependencies from root [`requirements.txt`](../requirements.txt) (`.[demo]`). Open **Logs** on first build if `pip install` fails.
5. Set the **app file** if you are not using the default `app.py`: in the Space README front matter, set `app_file` to `app/gradio_app.py` (see YAML below).
6. **Branch:** If the Space tracks a Git branch (or you mirror from GitHub), point it at this repository's **default branch**. Today that is `master`; if you rename the default branch to `main` (or anything else), update the Space settings so the tracked branch matches, or builds will lag behind the wrong ref.

## Public URL for the portfolio (embed / "Try live demo")

Use the **running app URL** (Hugging Face docs: embed / direct app URL), not the Space repository page.

**Preferred for CTAs and `NEXT_PUBLIC_SNAKE_DEMO_URL`:**

`https://<space-subdomain>.hf.space`

You can copy this from the Space UI (embed / direct app link). It opens the live Gradio app in one step.

**Less ideal for a "Try live demo" button** (repo/gallery page, not the app shell):

`https://huggingface.co/spaces/<user>/<space-name>`

In [mmaitland-portfolio](https://github.com/mmaitland300/mmaitland-portfolio), set:

`NEXT_PUBLIC_SNAKE_DEMO_URL=https://<space-subdomain>.hf.space`

That enables "Try live demo" and marks the Snake Detector project as operational in the UI.

### Portfolio copy note ("shipped" vs live demo)

If the site shows **shipped** when no demo URL is set, treat that as **case study / repo shipped**, not "live app is up," unless the surrounding copy makes that explicit. After you set `NEXT_PUBLIC_SNAKE_DEMO_URL`, the live CTA should use the **hf.space** URL above so visitors land on the app, not the Space repo page.

## Optional README YAML

```yaml
---
title: Snake Detector Demo
emoji: 🐍
colorFrom: gray
colorTo: green
sdk: gradio
app_file: app/gradio_app.py
pinned: false
---
```

Use `app_file: app.py` instead if you prefer the default root entry (thin re-export).

Adjust `sdk_version` only if the Space template requires a pinned Gradio version.

## Manual model fallback

If the release download is blocked, add `artifacts/model.joblib` via the Space **Files** tab (do not commit large binaries to git).

## Local check

```bash
pip install -e ".[demo]"
python app.py
# or (same module-level demo + bootstrap):
python app/gradio_app.py
```

Hugging Face imports the configured app file and uses `demo` only; `launch()` runs only when you execute the file as `__main__`.

## Handoff checklist (minimal path)

1. Make this repo Spaces-compatible (done: `demo` + bootstrap + [`requirements.txt`](../requirements.txt)).
2. Deploy to a Gradio CPU Space; confirm predict works.
3. Open the public **`https://<space-subdomain>.hf.space`** URL and sanity-check the UI.
4. Set **`NEXT_PUBLIC_SNAKE_DEMO_URL`** to that **hf.space** URL in Vercel and `.env.local` for the portfolio; redeploy the portfolio.

**One-line reminder:** Prefer `app_file: app/gradio_app.py` if you want all Space logic under `app/`; otherwise keep the default root `app.py`. Either way, expose **`demo = build_demo()`** at module scope and do not call **`launch()`** on import.

**Portfolio env:** Use the public Space **app** URL (`https://<subdomain>.hf.space`), not the `huggingface.co/spaces/...` gallery URL, for the live demo link.
