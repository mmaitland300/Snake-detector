from __future__ import annotations

from pathlib import Path

from .pipeline import predict_image


def run_predict(
    model_path: Path,
    image_path: Path,
    image_size: int,
    *,
    decision_threshold: float | None = None,
) -> tuple[str, float]:
    return predict_image(
        model_path=model_path,
        image_path=image_path,
        image_size=image_size,
        decision_threshold=decision_threshold,
    )
