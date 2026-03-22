from __future__ import annotations

from .config import AppConfig
from .pipeline import train_model


def run_train(config: AppConfig) -> dict:
    return train_model(config)
