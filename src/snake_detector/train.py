from __future__ import annotations

from pathlib import Path

from .config import AppConfig
from .pipeline import train_model


def run_train(config: AppConfig) -> dict:
    return train_model(config)


def default_model_path() -> Path:
    return Path("artifacts/model.keras")
