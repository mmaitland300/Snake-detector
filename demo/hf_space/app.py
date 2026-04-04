r"""
Gradio demo: same inference as the shipped model (InceptionV3 preprocess, resize, p_snake).
Visitor-facing labels use three bands (likely snake / undetermined / likely not); the project CLI stays binary with one threshold.

Local (repo): default deployment_config.json points at artifacts/... relative to this folder.
From demo/hf_space the venv is at the repo root (two levels up). Example:

  cd "C:\dev\Cursor Projects\Snake-detector\demo\hf_space"
  ..\..\.venv\Scripts\python app.py

Or run .\run_local.ps1 from hf_space.

In the browser use http://127.0.0.1:7860 (or the printed port). Do not open
http://0.0.0.0:7860 -- that is a listen address, not a valid URL on Windows (ERR_ADDRESS_INVALID).

Hugging Face: set model_path to "model.keras", upload model.keras, optional env:
  SNAKE_DETECTOR_THRESHOLD (high band), SNAKE_DETECTOR_LOW_THRESHOLD,
  SNAKE_DETECTOR_IMAGE_SIZE, SNAKE_DETECTOR_MODEL_PATH
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


def _low_threshold_from_env_or_config(cfg: dict, high: float) -> float:
    env = os.environ.get("SNAKE_DETECTOR_LOW_THRESHOLD", "").strip()
    if env:
        return float(env)
    if cfg.get("low_threshold") is not None:
        return float(cfg["low_threshold"])
    return round(1.0 - float(high), 2)


_CFG = _load_config()
if _CFG.get("preprocessing") != "inception_v3_preprocess_input":
    raise ValueError(
        "Only inception_v3_preprocess_input is supported in this demo "
        f"(got {_CFG.get('preprocessing')!r})."
    )

_MODEL_PATH = _resolve_model_path(_CFG)
_IMAGE_SIZE = _image_size_from_env_or_config(_CFG)
_THRESHOLD = _threshold_from_env_or_config(_CFG)
_LOW_THRESHOLD = _low_threshold_from_env_or_config(_CFG, _THRESHOLD)

if not 0.0 <= _THRESHOLD <= 1.0:
    raise ValueError(f"threshold must be in [0, 1], got {_THRESHOLD}")
if not 0.0 <= _LOW_THRESHOLD <= 1.0:
    raise ValueError(f"low_threshold must be in [0, 1], got {_LOW_THRESHOLD}")
if _LOW_THRESHOLD >= _THRESHOLD:
    raise ValueError(
        f"low_threshold ({_LOW_THRESHOLD}) must be < high threshold ({_THRESHOLD}) for tri-state UX"
    )

if not _MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model file not found: {_MODEL_PATH}. "
        "Fix deployment_config.json model_path, place model.keras next to app.py for HF, "
        "or set SNAKE_DETECTOR_MODEL_PATH to an absolute path."
    )

_MODEL = load_model(_MODEL_PATH)

# Dark, portfolio-adjacent card styling (cyan accent, restrained contrast).
_CUSTOM_CSS = """
.demo-hero { margin-bottom: 0.35rem !important; }
.demo-hero h1 {
  font-size: 1.35rem !important;
  font-weight: 650 !important;
  letter-spacing: -0.02em !important;
  margin: 0 0 0.35rem 0 !important;
}
.demo-tagline {
  font-size: 0.92rem !important;
  line-height: 1.45 !important;
  opacity: 0.78 !important;
  margin: 0 !important;
}
.section-label {
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  opacity: 0.55;
  margin: 0 0 0.45rem 0 !important;
}
.result-card {
  border-radius: 12px;
  border: 1px solid rgba(148, 163, 184, 0.22);
  background: rgba(15, 23, 42, 0.65);
  padding: 1.15rem 1.2rem;
  min-height: 7.5rem;
}
.result-card--idle {
  opacity: 0.85;
  display: flex;
  align-items: center;
}
.verdict {
  font-size: 1.28rem;
  font-weight: 700;
  line-height: 1.2;
  margin: 0 0 0.65rem 0;
  letter-spacing: -0.02em;
  padding-left: 0.65rem;
  border-left: 3px solid rgba(34, 211, 238, 0.55);
}
.verdict--likely { border-left-color: rgba(34, 211, 238, 0.85); }
.verdict--not { border-left-color: rgba(148, 163, 184, 0.5); }
.verdict--unsure { border-left-color: rgba(250, 204, 21, 0.65); }
.snake-pct {
  font-size: 2.05rem;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.03em;
  color: rgba(226, 232, 240, 0.98);
  margin: 0 0 0.15rem 0;
  line-height: 1.1;
}
.snake-pct-label {
  font-size: 0.8rem;
  opacity: 0.55;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 0.35rem;
}
.result-note {
  font-size: 0.9rem;
  line-height: 1.45;
  opacity: 0.82;
  margin: 0.75rem 0 0 0;
}
.tech-foot {
  margin-top: 0.9rem;
  padding-top: 0.65rem;
  border-top: 1px solid rgba(148, 163, 184, 0.15);
  font-size: 0.72rem;
  line-height: 1.4;
  opacity: 0.48;
}
.disclaimer {
  margin-top: 1.25rem;
  font-size: 0.75rem;
  line-height: 1.4;
  opacity: 0.42;
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
    hi, lo = _THRESHOLD, _LOW_THRESHOLD
    if p_snake >= hi:
        verdict = "likely_snake"
        verdict_text = "Likely a snake"
        note = "This score is in the demo's confident snake range."
    elif p_snake <= lo:
        verdict = "likely_not"
        verdict_text = "Likely not a snake"
        note = "This score is in the demo's confident not-snake range."
    else:
        verdict = "undetermined"
        verdict_text = "Undetermined"
        note = "The model is unsure; wide shots or tiny snakes are often hard to read."
    return verdict, {
        "verdict": verdict,
        "verdict_text": verdict_text,
        "note": note,
        "p_snake": p_snake,
        "threshold_high": hi,
        "threshold_low": lo,
        "image_size": _IMAGE_SIZE,
    }


def format_result_card(meta: dict) -> str:
    if not meta:
        return (
            '<div class="result-card result-card--idle">'
            "<p>Add a photo, then <strong>Predict</strong>.</p>"
            "</div>"
        )
    p_snake = float(meta["p_snake"])
    verdict = meta["verdict"]
    verdict_text = meta["verdict_text"]
    note = meta["note"]
    hi = float(meta["threshold_high"])
    lo = float(meta["threshold_low"])
    sz = int(meta["image_size"])
    disc = html.escape(DISCLAIMER)
    vcls = {
        "likely_snake": "verdict verdict--likely",
        "likely_not": "verdict verdict--not",
        "undetermined": "verdict verdict--unsure",
    }[verdict]
    hi_pct = int(round(hi * 100))
    lo_pct = int(round(lo * 100))
    tech = (
        f"Confident snake if estimate >= {hi_pct}%, confident not-snake if <= {lo_pct}%. "
        f"Between those values we show Undetermined. Photos are resized to {sz}x{sz}px before scoring."
    )
    return f"""<div class="result-card">
<p class="{vcls}">{html.escape(verdict_text)}</p>
<div class="snake-pct-label">Snake likelihood</div>
<div class="snake-pct">{p_snake:.0%}</div>
<p class="result-note">{html.escape(note)}</p>
<p class="tech-foot">{html.escape(tech)}</p>
<p class="disclaimer">{disc}</p>
</div>"""


def on_predict(image: Image.Image | None) -> str:
    _, meta = predict_image(image)
    return format_result_card(meta)


_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.zinc,
).set(
    body_background_fill="#09090b",
    block_background_fill="#121214",
    block_border_width="1px",
    border_color_primary="#27272a",
    body_text_color="#e4e4e7",
    block_label_text_color="#a1a1aa",
    input_background_fill="#18181b",
    button_primary_background_fill="#0891b2",
    button_primary_background_fill_hover="#06b6d4",
    button_primary_text_color="#fafafa",
)

with gr.Blocks(title="Snake detector (demo)") as demo:
    gr.Markdown(
        '<div class="demo-hero">\n'
        "# Snake in the photo?\n\n"
        '<p class="demo-tagline">Upload an image for a quick likelihood read, '
        "not a safety tool.</p></div>"
    )
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown('<p class="section-label">Photo</p>')
            inp = gr.Image(type="pil", label="Upload", height=260)
            predict_btn = gr.Button("Predict", variant="primary", size="lg")
        with gr.Column(scale=1):
            gr.Markdown('<p class="section-label">Result</p>')
            summary = gr.HTML(value=format_result_card({}))
    with gr.Accordion("How this works", open=False):
        gr.Markdown(
            "- **Snake likelihood** is one number from the model (higher means more snake-like).\n"
            "- **Likely a snake / Likely not a snake / Undetermined** uses two cutoffs on that number: "
            "confident calls only when the score is clearly high or clearly low; the middle band is "
            "shown as **Undetermined** so the UI does not overstate certainty.\n"
            "- **Wide shots:** the photo is shrunk to a small square, so a tiny snake can be missed. "
            "Closer, clearer photos work best.\n"
            "- The project **CLI** still uses a single threshold for binary snake / not-snake; this "
            "demo adds the extra band for visitors only.\n"
        )
    predict_btn.click(on_predict, inp, summary)

def _launch_server_name() -> str:
    # Spaces expose the app via proxy; bind all interfaces. Locally, 0.0.0.0 is not a usable browser URL.
    if os.environ.get("SPACE_ID", "").strip():
        return "0.0.0.0"
    return "127.0.0.1"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(
        server_name=_launch_server_name(),
        server_port=port,
        theme=_theme,
        css=_CUSTOM_CSS,
    )
