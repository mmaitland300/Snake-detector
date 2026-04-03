r"""
Gradio demo: same frozen behavior as CLI (InceptionV3 preprocess, threshold, image size).

Local (repo): default deployment_config.json points at artifacts/... relative to this folder.
From demo/hf_space the venv is at the repo root (two levels up). Example:

  cd "C:\dev\Cursor Projects\Snake-detector\demo\hf_space"
  ..\..\.venv\Scripts\python app.py

Or run .\run_local.ps1 from hf_space.

In the browser use http://127.0.0.1:7860 (or the printed port). Do not open
http://0.0.0.0:7860 -- that is a listen address, not a valid URL on Windows (ERR_ADDRESS_INVALID).

Hugging Face: set model_path to "model.keras", upload model.keras, optional env:
  SNAKE_DETECTOR_THRESHOLD, SNAKE_DETECTOR_IMAGE_SIZE, SNAKE_DETECTOR_MODEL_PATH
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model

_APP_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _APP_DIR / "deployment_config.json"

DISCLAIMER = (
    "For demonstration only. Not a safety device. "
    "Do not use for medical, legal, or wildlife handling decisions."
)


def _load_config() -> dict:
    with _CONFIG_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_model_path(cfg: dict) -> Path:
    raw = os.environ.get("SNAKE_DETECTOR_MODEL_PATH", "").strip() or cfg["model_path"]
    path = Path(raw)
    if not path.is_absolute():
        path = (_CONFIG_PATH.parent / path).resolve()
    return path


def _threshold_from_env_or_config(cfg: dict) -> float:
    env = os.environ.get("SNAKE_DETECTOR_THRESHOLD", "").strip()
    if env:
        return float(env)
    return float(cfg["threshold"])


def _image_size_from_env_or_config(cfg: dict) -> int:
    env = os.environ.get("SNAKE_DETECTOR_IMAGE_SIZE", "").strip()
    if env:
        return int(env)
    return int(cfg["image_size"])


_CFG = _load_config()
if _CFG.get("preprocessing") != "inception_v3_preprocess_input":
    raise ValueError(
        "Only inception_v3_preprocess_input is supported in this demo "
        f"(got {_CFG.get('preprocessing')!r})."
    )

_MODEL_PATH = _resolve_model_path(_CFG)
_IMAGE_SIZE = _image_size_from_env_or_config(_CFG)
_THRESHOLD = _threshold_from_env_or_config(_CFG)

if not 0.0 <= _THRESHOLD <= 1.0:
    raise ValueError(f"threshold must be in [0, 1], got {_THRESHOLD}")

if not _MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model file not found: {_MODEL_PATH}. "
        "Fix deployment_config.json model_path, place model.keras next to app.py for HF, "
        "or set SNAKE_DETECTOR_MODEL_PATH to an absolute path."
    )

_MODEL = load_model(_MODEL_PATH)

# Scoped UI styling (works with Gradio light/dark via theme variables where available).
_CUSTOM_CSS = """
.demo-lead { opacity: 0.92; margin-bottom: 0.25rem !important; }
.section-title {
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  opacity: 0.65;
  margin: 0 0 0.5rem 0 !important;
}
.result-card {
  border-radius: 14px;
  border: 1px solid var(--border-color-primary, #e5e7eb);
  background: var(--background-fill-secondary, #f9fafb);
  padding: 1.25rem 1.35rem;
  min-height: 8rem;
}
.result-card--idle {
  opacity: 0.88;
  display: flex;
  align-items: center;
}
.verdict {
  font-size: 1.35rem;
  font-weight: 700;
  line-height: 1.25;
  margin: 0 0 0.85rem 0;
  letter-spacing: -0.02em;
}
.metrics { margin: 0; padding: 0; list-style: none; }
.metrics li {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.4rem 0;
  border-bottom: 1px solid var(--border-color-primary, #e5e7eb);
  font-size: 0.98rem;
}
.metrics li:last-child { border-bottom: none; }
.metrics .pct { font-variant-numeric: tabular-nums; font-weight: 600; }
.rule {
  margin-top: 1rem;
  padding: 0.75rem 0.9rem;
  border-radius: 10px;
  background: var(--background-fill-primary, #fff);
  border: 1px solid var(--border-color-primary, #e5e7eb);
  font-size: 0.92rem;
  line-height: 1.45;
}
.disclaimer {
  margin-top: 1rem;
  font-size: 0.8rem;
  line-height: 1.4;
  opacity: 0.72;
}
"""


def predict_image(pil_img: Image.Image | None) -> tuple[str, dict]:
    if pil_img is None:
        return "", {}
    rgb = pil_img.convert("RGB").resize((_IMAGE_SIZE, _IMAGE_SIZE))
    arr = np.asarray(rgb, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    p_snake = float(_MODEL.predict(arr, verbose=0)[0][0])
    p_not_snake = 1.0 - p_snake
    label = "snake" if p_snake >= _THRESHOLD else "no_snake"
    return label, {
        "label": label,
        "p_snake": p_snake,
        "p_not_snake": p_not_snake,
        "threshold": _THRESHOLD,
        "image_size": _IMAGE_SIZE,
    }


def format_result_card(meta: dict) -> str:
    if not meta:
        return (
            '<div class="result-card result-card--idle">'
            "<p>Choose a photo above, then tap <strong>Predict</strong>.</p>"
            "</div>"
        )
    p_snake = float(meta["p_snake"])
    p_not_snake = float(meta["p_not_snake"])
    label = meta["label"]
    verdict = "Likely a snake" if label == "snake" else "Likely not a snake"
    threshold_pct = int(round(float(meta["threshold"]) * 100))
    sz = int(meta["image_size"])
    disc = html.escape(DISCLAIMER)
    return f"""<div class="result-card">
<p class="verdict">{html.escape(verdict)}</p>
<ul class="metrics">
<li><span>Snake probability</span><span class="pct">{p_snake:.1%}</span></li>
<li><span>Not-snake probability</span><span class="pct">{p_not_snake:.1%}</span></li>
</ul>
<p class="rule">
This demo calls an image <strong>snake</strong> when snake probability is at least
<strong>{threshold_pct}%</strong>. Photos are resized to <strong>{sz} x {sz}</strong> pixels before scoring.
</p>
<p class="disclaimer">{disc}</p>
</div>"""


def format_scores(meta: dict) -> dict[str, float]:
    if not meta:
        return {}
    return {
        "Snake": round(float(meta["p_snake"]), 4),
        "Not a snake": round(float(meta["p_not_snake"]), 4),
    }


def on_predict(image: Image.Image | None) -> tuple[dict[str, float], str]:
    _, meta = predict_image(image)
    return format_scores(meta), format_result_card(meta)


with gr.Blocks(
    title="Snake vs non-snake (demo)",
    theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="slate"),
    css=_CUSTOM_CSS,
) as demo:
    gr.Markdown(
        "# Is there a snake in this photo?\n\n"
        '<p class="demo-lead">Upload a picture. The model estimates snake vs not-snake '
        "probabilities; the plain-English line is just how we apply the demo's cutoff.</p>"
    )
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown('<p class="section-title">1. Your photo</p>')
            inp = gr.Image(type="pil", label="Upload", height=280)
            predict_btn = gr.Button("Predict", variant="primary", size="lg")
        with gr.Column(scale=1):
            gr.Markdown('<p class="section-title">2. Scores (model output)</p>')
            scores = gr.Label(label="Relative confidence", num_top_classes=2)
            gr.Markdown('<p class="section-title">3. Plain-English summary</p>')
            summary = gr.HTML(value=format_result_card({}))
    gr.Markdown(
        "### How to read this\n\n"
        "- **Scores** are the model's two outputs (they add up to 100%).\n"
        "- **Likely a snake / Likely not a snake** uses a fixed cutoff from the deployment config "
        "(same rule as the project CLI when thresholds match).\n"
        "- **Wide or zoomed-out photos:** the image is resized to a small square before scoring, so a "
        "snake that only covers a few pixels can be easy to miss. The training data skews toward "
        "clearer, closer views. Bounding-box detection is a separate future step, not this demo.\n"
    )
    predict_btn.click(on_predict, inp, [scores, summary])

def _launch_server_name() -> str:
    # Spaces expose the app via proxy; bind all interfaces. Locally, 0.0.0.0 is not a usable browser URL.
    if os.environ.get("SPACE_ID", "").strip():
        return "0.0.0.0"
    return "127.0.0.1"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name=_launch_server_name(), server_port=port)
