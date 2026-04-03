from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class DataConfig:
    raw_dir: Path = Path("data/raw")
    split_dir: Path = Path("data/split")
    train_split: float = 0.8
    val_split: float = 0.1
    seed: int = 42


@dataclass(slots=True)
class TrainConfig:
    image_size: int = 150
    batch_size: int = 20
    epochs: int = 10
    learning_rate: float = 1e-4
    backbone: str = "inceptionv3"
    freeze_backbone: bool = True
    # Balanced weights from training folder counts (Keras class_weight; sklearn sample_weight).
    use_class_weights: bool = True


@dataclass(slots=True)
class InferenceConfig:
    """Binary decision rule: predict snake if P(snake) >= decision_threshold."""

    decision_threshold: float = 0.5


@dataclass(slots=True)
class PathsConfig:
    model_path: Path = Path("artifacts/model.keras")
    metrics_path: Path = Path("artifacts/metrics.json")
    confusion_matrix_path: Path = Path("artifacts/confusion_matrix.png")
    predictions_panel_path: Path = Path("artifacts/sample_predictions.png")
    predictions_manifest_path: Path = Path("artifacts/sample_predictions.json")


@dataclass(slots=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_config(config: AppConfig, output_path: Path) -> None:
    ensure_parent(output_path)
    payload = asdict(config)
    payload["data"]["raw_dir"] = str(config.data.raw_dir)
    payload["data"]["split_dir"] = str(config.data.split_dir)
    payload["paths"]["model_path"] = str(config.paths.model_path)
    payload["paths"]["metrics_path"] = str(config.paths.metrics_path)
    payload["paths"]["confusion_matrix_path"] = str(config.paths.confusion_matrix_path)
    payload["paths"]["predictions_panel_path"] = str(config.paths.predictions_panel_path)
    payload["paths"]["predictions_manifest_path"] = str(config.paths.predictions_manifest_path)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
