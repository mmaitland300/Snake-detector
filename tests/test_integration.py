"""End-to-end integration test: generate → split → train → eval → predict."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from snake_detector.cli import main as cli_main
from snake_detector.demo_data import generate_demo_dataset


@pytest.fixture()
def pipeline_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    generate_demo_dataset(output_dir=raw, samples_per_class=20, image_size=32, seed=7)
    return tmp_path


def test_full_sklearn_pipeline(pipeline_dir: Path) -> None:
    raw = pipeline_dir / "raw"
    split = pipeline_dir / "split"
    model = pipeline_dir / "model.joblib"
    metrics = pipeline_dir / "metrics.json"

    assert cli_main(["split", "--raw-dir", str(raw), "--split-dir", str(split), "--seed", "7"]) == 0
    assert (split / "training").exists()
    assert (split / "testing").exists()

    assert cli_main([
        "train",
        "--split-dir", str(split),
        "--model-path", str(model),
        "--metrics-path", str(metrics),
        "--image-size", "32",
        "--batch-size", "4",
        "--epochs", "2",
        "--seed", "7",
        "--backbone", "sklearn_mlp",
    ]) == 0
    assert model.exists()
    assert metrics.exists()

    data = json.loads(metrics.read_text(encoding="utf-8"))
    for key in (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "confusion_matrix",
        "decision_threshold",
    ):
        assert key in data, f"Missing metric key: {key}"

    assert cli_main([
        "eval",
        "--split-dir", str(split),
        "--model-path", str(model),
        "--metrics-path", str(metrics),
        "--image-size", "32",
        "--batch-size", "4",
        "--backbone", "sklearn_mlp",
    ]) == 0

    test_images = list((split / "testing").rglob("*.png"))
    assert len(test_images) > 0
    assert cli_main([
        "predict",
        "--model-path", str(model),
        "--image-path", str(test_images[0]),
        "--image-size", "32",
    ]) == 0
