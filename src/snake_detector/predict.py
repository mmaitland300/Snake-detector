from __future__ import annotations

from pathlib import Path

from .pipeline import predict_image


def run_predict(model_path: Path, image_path: Path, image_size: int) -> tuple[str, float]:
    return predict_image(model_path=model_path, image_path=image_path, image_size=image_size)
