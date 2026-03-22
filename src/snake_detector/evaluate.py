from __future__ import annotations

from .config import AppConfig
from .pipeline import evaluate_saved_model


def run_evaluate(config: AppConfig) -> dict:
    return evaluate_saved_model(config)
