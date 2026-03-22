from __future__ import annotations

from pathlib import Path

from PIL import Image

from snake_detector.pipeline import predict_image, resolve_prediction_image_size

MODEL_PATH_CANDIDATES = [Path("artifacts/model.joblib"), Path("artifacts/model.keras")]


def _find_model_path() -> Path | None:
    for candidate in MODEL_PATH_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _predict(image: Image.Image) -> dict[str, float | str]:
    if image is None:
        return {"label": "no_image", "confidence": 0.0}
    model_path = _find_model_path()
    if model_path is None:
        return {
            "label": "model_missing",
            "confidence": 0.0,
            "message": "No model found at artifacts/model.joblib or artifacts/model.keras. Train or provide a model artifact.",
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


def build_demo():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Gradio is required. Install with `pip install .[demo]`.") from exc

    with gr.Blocks(title="Snake Detector Demo") as demo:
        gr.Markdown(
            "# Snake Detector\n"
            "Upload an image to classify as `snake` or `no_snake`.\n\n"
            "Portfolio-safe demo mode defaults to a locally generated placeholder model artifact.\n"
            "Free-tier deployments may take a moment to wake up."
        )
        image_input = gr.Image(type="pil", label="Input image")
        output = gr.JSON(label="Prediction")
        run_button = gr.Button("Predict")
        run_button.click(fn=_predict, inputs=[image_input], outputs=[output])
    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch()
