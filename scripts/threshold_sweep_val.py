from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snake_detector.data import RAW_IMAGE_EXTENSIONS
from snake_detector.pipeline import resolve_prediction_image_size


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep classification thresholds on the validation split for a saved Keras model."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/collection_main_v1/model.keras"),
        help="Path to the saved Keras model.",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path("data/split_collection_main_v1/validation"),
        help="Directory containing validation class folders.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/collection_main_v1/threshold_sweep_val.json"),
        help="Where to save the threshold sweep JSON.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=0,
        help="Resize dimension. If omitted or 0, infer from run_config/model metadata.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for validation prediction.")
    parser.add_argument(
        "--threshold-start",
        type=float,
        default=0.05,
        help="Starting threshold, inclusive.",
    )
    parser.add_argument(
        "--threshold-stop",
        type=float,
        default=0.95,
        help="Ending threshold, inclusive when aligned with the step.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Threshold step size.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.threshold_step <= 0:
        raise SystemExit("--threshold-step must be > 0")
    if not (0.0 <= args.threshold_start <= 1.0 and 0.0 <= args.threshold_stop <= 1.0):
        raise SystemExit("threshold bounds must be within [0, 1]")
    if args.threshold_start > args.threshold_stop:
        raise SystemExit("--threshold-start must be <= --threshold-stop")


def threshold_grid(start: float, stop: float, step: float) -> list[float]:
    count = int(math.floor((stop - start) / step)) + 1
    values = [round(start + step * idx, 10) for idx in range(count)]
    if values[-1] < stop and (stop - values[-1]) > 1e-9:
        values.append(round(stop, 10))
    return values


def load_validation_records(validation_dir: Path) -> tuple[list[Path], np.ndarray, dict[str, int]]:
    class_dirs = sorted(p for p in validation_dir.iterdir() if p.is_dir())
    if len(class_dirs) < 2:
        raise SystemExit(f"Expected at least two class directories under {validation_dir}")

    class_indices = {class_dir.name: idx for idx, class_dir in enumerate(class_dirs)}
    image_paths: list[Path] = []
    labels: list[int] = []
    for class_dir in class_dirs:
        idx = class_indices[class_dir.name]
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in RAW_IMAGE_EXTENSIONS:
                image_paths.append(image_path)
                labels.append(idx)

    if not image_paths:
        raise SystemExit(f"No validation images found under {validation_dir}")
    return image_paths, np.asarray(labels, dtype=np.int32), class_indices


def predict_scores(model_path: Path, image_paths: list[Path], image_size: int, batch_size: int) -> np.ndarray:
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import img_to_array, load_img

    model = load_model(model_path)
    scores: list[np.ndarray] = []
    for offset in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[offset : offset + batch_size]
        batch = []
        for image_path in batch_paths:
            img = load_img(image_path, target_size=(image_size, image_size), color_mode="rgb")
            arr = img_to_array(img)
            batch.append(arr)
        batch_arr = preprocess_input(np.stack(batch, axis=0).astype(np.float32))
        batch_scores = model.predict(batch_arr, verbose=0).flatten()
        scores.append(batch_scores)
    return np.concatenate(scores, axis=0)


def metric_row(
    *,
    threshold: float,
    y_true: np.ndarray,
    scores: np.ndarray,
    positive_index: int,
) -> dict[str, float | int]:
    y_pred = (scores >= threshold).astype(int)

    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    neg_index = 1 - positive_index
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "snake_precision": round(float(precision[positive_index]), 6),
        "snake_recall": round(float(recall[positive_index]), 6),
        "snake_f1": round(float(f1[positive_index]), 6),
        "no_snake_precision": round(float(precision[neg_index]), 6),
        "no_snake_recall": round(float(recall[neg_index]), 6),
        "no_snake_f1": round(float(f1[neg_index]), 6),
        "macro_precision": round(float(macro_precision), 6),
        "macro_recall": round(float(macro_recall), 6),
        "macro_f1": round(float(macro_f1), 6),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "false_positive_rate": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 6),
    }


def pick_best(rows: list[dict[str, float | int]], key: str) -> dict[str, float | int]:
    return max(rows, key=lambda row: (float(row[key]), -abs(float(row["threshold"]) - 0.5)))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    model_path = args.model_path
    validation_dir = args.validation_dir
    output_path = args.output_path

    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not validation_dir.exists():
        raise SystemExit(f"Validation directory not found: {validation_dir}")

    image_size = args.image_size if args.image_size > 0 else resolve_prediction_image_size(model_path)
    image_paths, y_true, class_indices = load_validation_records(validation_dir)
    if "snake" not in class_indices:
        raise SystemExit("Validation split does not contain a 'snake' class directory.")

    positive_index = int(class_indices["snake"])
    scores = predict_scores(model_path, image_paths, image_size=image_size, batch_size=args.batch_size)

    rows = [
        metric_row(threshold=value, y_true=y_true, scores=scores, positive_index=positive_index)
        for value in threshold_grid(args.threshold_start, args.threshold_stop, args.threshold_step)
    ]

    baseline = next((row for row in rows if abs(float(row["threshold"]) - 0.5) < 1e-9), None)
    if baseline is None:
        baseline = metric_row(threshold=0.5, y_true=y_true, scores=scores, positive_index=positive_index)
        rows.append(baseline)
        rows.sort(key=lambda row: float(row["threshold"]))

    best_snake_recall = pick_best(rows, "snake_recall")
    best_snake_f1 = pick_best(rows, "snake_f1")
    best_macro_f1 = pick_best(rows, "macro_f1")

    payload = {
        "model_path": str(model_path),
        "validation_dir": str(validation_dir),
        "output_path": str(output_path),
        "image_size": image_size,
        "batch_size": args.batch_size,
        "class_indices": {name: int(idx) for name, idx in class_indices.items()},
        "samples": int(len(image_paths)),
        "baseline_threshold_0_5": baseline,
        "recommended": {
            "best_snake_recall": best_snake_recall,
            "best_snake_f1": best_snake_f1,
            "best_macro_f1": best_macro_f1,
        },
        "rows": rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload["recommended"], indent=2))
    print(f"\nSaved threshold sweep to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
