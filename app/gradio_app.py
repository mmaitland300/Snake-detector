from __future__ import annotations

import importlib.util
import logging
import types
from pathlib import Path
from typing import Any

from PIL import Image

from snake_detector.pipeline import predict_image, resolve_prediction_image_size

logger = logging.getLogger(__name__)

# Keep in sync with `app/hf_bootstrap._MIN_BYTES` (minimum artifact size to trust on disk).
_MIN_ARTIFACT_BYTES = 512

MODEL_PATH_CANDIDATES = [Path("artifacts/model.joblib"), Path("artifacts/model.keras")]


def _find_model_path() -> Path | None:
    for candidate in MODEL_PATH_CANDIDATES:
        if candidate.is_file() and candidate.stat().st_size >= _MIN_ARTIFACT_BYTES:
            return candidate
    return None


def _predict(image: Image.Image) -> dict[str, float | str]:
    if image is None:
        return {"label": "no_image", "confidence": 0.0}
    model_path = _find_model_path()
    if model_path is None:
        msg = (
            "No model found at artifacts/model.joblib or artifacts/model.keras. "
            "Train or provide a model artifact."
        )
        if _BOOTSTRAP_LAST_ERROR:
            msg += (
                f" Automatic download did not complete ({_BOOTSTRAP_LAST_ERROR}). "
                "On Hugging Face Spaces, upload model.joblib under artifacts/ via the Files tab, "
                "then restart the Space if predictions still fail."
            )
        return {
            "label": "model_missing",
            "confidence": 0.0,
            "message": msg,
        }

    temp_path = Path("artifacts/.demo_input.png")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(temp_path)
    image_size = resolve_prediction_image_size(model_path)
    label, confidence = predict_image(model_path, temp_path, image_size=image_size)
    return {
        "label": label,
        "confidence": round(float(confidence), 4),
        "model_path": str(model_path),
        "image_size": image_size,
    }


def build_demo() -> Any:
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Gradio is required. Install with `pip install .[demo]`.") from exc

    with gr.Blocks(title="Snake Detector Demo") as demo:
        gr.Markdown(
            "# Snake Detector (bounded demo)\n\n"
            "Binary experiment: label an upload as `snake` or `no_snake`. "
            "This is **not** species identification or field-ready wildlife classification.\n\n"
            "The published model is a **pipeline-validation** artifact trained on generated "
            "placeholder imagery; see the repository README for benchmark scope and legal boundaries.\n\n"
            "Free-tier hosts may sleep briefly on first open."
        )
        image_input = gr.Image(type="pil", label="Input image")
        output = gr.JSON(label="Prediction")
        run_button = gr.Button("Predict")
        run_button.click(fn=_predict, inputs=[image_input], outputs=[output])
    return demo


def _load_hf_bootstrap() -> types.ModuleType:
    bootstrap = Path(__file__).resolve().parent / "hf_bootstrap.py"
    spec = importlib.util.spec_from_file_location("snake_hf_bootstrap", bootstrap)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {bootstrap}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_BOOTSTRAP_LAST_ERROR: str | None = None
try:
    _load_hf_bootstrap()._ensure_model()
except Exception as exc:
    _BOOTSTRAP_LAST_ERROR = f"{type(exc).__name__}: {exc}"
    logger.warning(
        "Model bootstrap failed; serving UI without a downloaded artifact: %s",
        exc,
        exc_info=True,
    )

demo = build_demo()

if __name__ == "__main__":
    demo.launch()
