from __future__ import annotations

import json
from pathlib import Path

from .config import AppConfig
from .pipeline import evaluate_saved_model


def run_evaluate(config: AppConfig) -> dict:
    return evaluate_saved_model(config)


def write_metrics_markdown(metrics_path: Path, output_path: Path) -> None:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    table = (
        "| Metric | Value |\n"
        "| --- | ---: |\n"
        f"| Accuracy | {metrics.get('accuracy', 0):.4f} |\n"
        f"| Macro Precision | {metrics.get('macro_precision', 0):.4f} |\n"
        f"| Macro Recall | {metrics.get('macro_recall', 0):.4f} |\n"
        f"| Macro F1 | {metrics.get('macro_f1', 0):.4f} |\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table, encoding="utf-8")
