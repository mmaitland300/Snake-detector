from __future__ import annotations

import json
import os
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

from .config import AppConfig, ensure_parent, save_config
from .data import build_generators
from .models import ModelSpec, build_binary_classifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def train_model(config: AppConfig) -> dict:
    set_seed(config.data.seed)
    if config.train.backbone.lower() == "sklearn_mlp":
        return _train_sklearn_model(config)

    train_gen, val_gen, test_gen = build_generators(
        split_dir=config.data.split_dir,
        image_size=config.train.image_size,
        batch_size=config.train.batch_size,
        seed=config.data.seed,
    )
    model = build_binary_classifier(
        ModelSpec(
            backbone=config.train.backbone,
            image_size=config.train.image_size,
            learning_rate=config.train.learning_rate,
            freeze_backbone=config.train.freeze_backbone,
        )
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.train.epochs,
    )
    ensure_parent(config.paths.model_path)
    model.save(config.paths.model_path)

    metrics = evaluate_model(model, test_gen, config)
    metrics["history"] = history.history
    _save_training_curves(history.history, config.paths.metrics_path.parent / "training_curves.png")
    save_config(config, config.paths.metrics_path.parent / "run_config.json")
    return metrics


def evaluate_saved_model(config: AppConfig) -> dict:
    artifact = _maybe_load_joblib_artifact(config.paths.model_path)
    if artifact is not None:
        test_images, test_labels, class_names = _load_split_records(config.data.split_dir / "testing")
        x_test = np.stack([_extract_feature_vector(path, artifact["feature_size"]) for path in test_images])
        scores = artifact["classifier"].predict_proba(x_test)[:, 1]
        pred_labels = artifact["classifier"].predict(x_test)
        return _save_evaluation_outputs(
            true_labels=np.array(test_labels),
            pred_labels=np.array(pred_labels),
            scores=np.array(scores),
            class_names=class_names,
            image_paths=test_images,
            config=config,
        )

    try:
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required. Install with `pip install .[ml]`.") from exc

    _, _, test_gen = build_generators(
        split_dir=config.data.split_dir,
        image_size=config.train.image_size,
        batch_size=config.train.batch_size,
        seed=config.data.seed,
    )
    model = load_model(config.paths.model_path)
    return evaluate_model(model, test_gen, config)


def evaluate_model(model, test_gen, config: AppConfig) -> dict:
    test_gen.reset()
    raw_preds = model.predict(test_gen)
    scores = raw_preds.flatten()
    pred_labels = (scores >= 0.5).astype(int)
    true_labels = test_gen.classes
    image_paths = [Path(test_gen.directory) / filename for filename in test_gen.filenames]
    return _save_evaluation_outputs(
        true_labels=true_labels,
        pred_labels=pred_labels,
        scores=scores,
        class_names=list(test_gen.class_indices.keys()),
        image_paths=image_paths,
        config=config,
    )


def resolve_prediction_image_size(model_path: Path, *, default: int = 150) -> int:
    """Spatial size to resize inputs for Keras inference.

    Order: env ``SNAKE_DETECTOR_IMAGE_SIZE``, ``run_config.json`` next to the model,
    then Keras ``model.input_shape``. Joblib models ignore this at inference
    (features use ``artifact['feature_size']``) but a consistent value is still
    returned for logging/UI.
    """
    env_raw = os.environ.get("SNAKE_DETECTOR_IMAGE_SIZE", "").strip()
    if env_raw:
        try:
            parsed = int(env_raw)
            if parsed > 0:
                return parsed
        except ValueError:
            pass

    if _maybe_load_joblib_artifact(model_path) is not None:
        run_size = _read_run_config_image_size(model_path.parent / "run_config.json")
        return run_size if run_size is not None else default

    keras_size = _read_run_config_image_size(model_path.parent / "run_config.json")
    if keras_size is not None:
        return keras_size

    return _infer_keras_input_spatial_size(model_path, default=default)


def _read_run_config_image_size(run_config_path: Path) -> int | None:
    if not run_config_path.exists():
        return None
    try:
        data = json.loads(run_config_path.read_text(encoding="utf-8"))
        size = int(data.get("train", {}).get("image_size", 0))
        return size if size > 0 else None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _infer_keras_input_spatial_size(model_path: Path, *, default: int) -> int:
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        return default
    try:
        model = load_model(model_path)
    except Exception:
        return default
    shape = model.input_shape
    if not shape or len(shape) < 4:
        return default
    # channels_last NHWC: (None, H, W, C)
    if shape[-1] in (1, 3, 4) or shape[-1] is None:
        h, w = shape[1], shape[2]
    else:
        # channels_first NCHW
        h, w = shape[2], shape[3]
    if h is not None and w is not None and int(h) == int(w):
        return int(h)
    return default


def predict_image(model_path: Path, image_path: Path, image_size: int = 150) -> tuple[str, float]:
    artifact = _maybe_load_joblib_artifact(model_path)
    if artifact is not None:
        vector = _extract_feature_vector(image_path, artifact["feature_size"])
        probabilities = artifact["classifier"].predict_proba(vector.reshape(1, -1))[0]
        class_names = artifact["class_names"]
        pred_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[pred_idx])
        return class_names[pred_idx], confidence

    try:
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required. Install with `pip install .[ml]`.") from exc

    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    arr = np.asarray(image, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    model = load_model(model_path)
    confidence = float(model.predict(arr)[0][0])
    label = "snake" if confidence >= 0.5 else "no_snake"
    display_confidence = confidence if label == "snake" else 1.0 - confidence
    return label, float(display_confidence)


def build_sample_predictions_panel(
    samples: list[tuple[Path, str, float]],
    output_path: Path,
    thumb_size: int = 220,
) -> None:
    if not samples:
        return

    cols = min(4, len(samples))
    rows = int(np.ceil(len(samples) / cols))
    caption_height = 44
    panel = Image.new("RGB", (cols * thumb_size, rows * (thumb_size + caption_height)), "white")
    draw = ImageDraw.Draw(panel)

    for idx, (path, label, confidence) in enumerate(samples):
        img = Image.open(path).convert("RGB")
        tile = ImageOps.fit(img, (thumb_size, thumb_size))
        x = (idx % cols) * thumb_size
        y = (idx // cols) * (thumb_size + caption_height)
        panel.paste(tile, (x, y))
        caption = f"{label} ({confidence:.2f})"
        draw.text((x + 8, y + thumb_size + 12), caption, fill="black")

    ensure_parent(output_path)
    panel.save(output_path)


def _save_training_curves(history: dict, output_path: Path) -> None:
    plt.style.use("ggplot")
    plt.figure()
    epochs = range(1, len(history.get("loss", [])) + 1)
    plt.plot(epochs, history.get("loss", []), label="train_loss")
    plt.plot(epochs, history.get("val_loss", []), label="val_loss")
    plt.plot(epochs, history.get("accuracy", []), label="train_accuracy")
    plt.plot(epochs, history.get("val_accuracy", []), label="val_accuracy")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend(loc="lower left")
    ensure_parent(output_path)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _train_sklearn_model(config: AppConfig) -> dict:
    train_images, train_labels, class_names = _load_split_records(config.data.split_dir / "training")
    test_images, test_labels, _ = _load_split_records(config.data.split_dir / "testing")
    feature_size = min(config.train.image_size, 48)

    x_train = np.stack([_extract_feature_vector(path, feature_size) for path in train_images])
    x_test = np.stack([_extract_feature_vector(path, feature_size) for path in test_images])

    classifier = SklearnPipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    learning_rate_init=config.train.learning_rate,
                    max_iter=max(80, config.train.epochs * 25),
                    early_stopping=True,
                    random_state=config.data.seed,
                ),
            ),
        ]
    )
    classifier.fit(x_train, train_labels)

    artifact = {
        "backend": "sklearn_mlp",
        "feature_size": feature_size,
        "class_names": class_names,
        "classifier": classifier,
    }
    ensure_parent(config.paths.model_path)
    joblib.dump(artifact, config.paths.model_path)

    scores = classifier.predict_proba(x_test)[:, 1]
    pred_labels = classifier.predict(x_test)
    metrics = _save_evaluation_outputs(
        true_labels=np.array(test_labels),
        pred_labels=np.array(pred_labels),
        scores=np.array(scores),
        class_names=class_names,
        image_paths=test_images,
        config=config,
    )
    metrics["history"] = {
        "backend": "sklearn_mlp",
        "iterations": int(classifier.named_steps["mlp"].n_iter_),
        "loss_curve": [float(loss) for loss in classifier.named_steps["mlp"].loss_curve_],
    }
    save_config(config, config.paths.metrics_path.parent / "run_config.json")
    return metrics


def _save_evaluation_outputs(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    scores: np.ndarray,
    class_names: list[str],
    image_paths: list[Path],
    config: AppConfig,
) -> dict:
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, pred_labels).tolist()

    ensure_parent(config.paths.confusion_matrix_path)
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(matrix), display_labels=class_names)
    disp.plot(colorbar=False)
    disp.figure_.savefig(config.paths.confusion_matrix_path, bbox_inches="tight")
    plt.close(disp.figure_)

    sample_manifest = _select_sample_predictions(
        image_paths=image_paths,
        class_names=class_names,
        true_labels=true_labels,
        pred_labels=pred_labels,
        scores=scores,
    )
    build_sample_predictions_panel(
        [
            (Path(sample["image_path"]), sample["prediction"], float(sample["confidence"]))
            for sample in sample_manifest
        ],
        config.paths.predictions_panel_path,
    )
    ensure_parent(config.paths.predictions_manifest_path)
    config.paths.predictions_manifest_path.write_text(
        json.dumps(sample_manifest, indent=2),
        encoding="utf-8",
    )

    metrics = {
        "accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": matrix,
        "class_names": class_names,
        "confusion_matrix_path": str(config.paths.confusion_matrix_path),
        "predictions_panel_path": str(config.paths.predictions_panel_path),
        "predictions_manifest_path": str(config.paths.predictions_manifest_path),
    }
    ensure_parent(config.paths.metrics_path)
    config.paths.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _load_split_records(split_root: Path) -> tuple[list[Path], list[int], list[str]]:
    class_names = sorted(path.name for path in split_root.iterdir() if path.is_dir())
    image_paths: list[Path] = []
    labels: list[int] = []
    for label_idx, class_name in enumerate(class_names):
        class_dir = split_root / class_name
        for image_path in sorted(class_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            image_paths.append(image_path)
            labels.append(label_idx)
    return image_paths, labels, class_names


def _extract_feature_vector(image_path: Path, feature_size: int) -> np.ndarray:
    image = Image.open(image_path).convert("L").resize((feature_size, feature_size))
    return np.asarray(image, dtype=np.float32).reshape(-1) / 255.0


def _maybe_load_joblib_artifact(model_path: Path) -> dict | None:
    try:
        artifact = joblib.load(model_path)
    except Exception:
        return None
    if isinstance(artifact, dict) and "classifier" in artifact and "class_names" in artifact:
        return artifact
    return None


def _select_sample_predictions(
    image_paths: list[Path],
    class_names: list[str],
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    scores: np.ndarray,
) -> list[dict]:
    manifest: list[dict] = []
    groups: dict[str, list[dict]] = {
        "correct_snake": [],
        "correct_no_snake": [],
        "failure": [],
    }

    for image_path, truth, pred, score in zip(image_paths, true_labels, pred_labels, scores, strict=True):
        predicted_label = class_names[int(pred)]
        truth_label = class_names[int(truth)]
        confidence = float(score if predicted_label == "snake" else 1.0 - score)
        record = {
            "image_path": str(image_path),
            "ground_truth": truth_label,
            "prediction": predicted_label,
            "confidence": round(confidence, 4),
            "selection_reason": "",
        }
        if truth == pred and predicted_label == "snake":
            groups["correct_snake"].append(record)
        elif truth == pred and predicted_label == "no_snake":
            groups["correct_no_snake"].append(record)
        else:
            groups["failure"].append(record)

    groups["correct_snake"].sort(key=lambda item: item["confidence"], reverse=True)
    groups["correct_no_snake"].sort(key=lambda item: item["confidence"], reverse=True)
    groups["failure"].sort(key=lambda item: abs(0.5 - item["confidence"]))

    for key, needed, reason in (
        ("correct_snake", 2, "correct snake prediction"),
        ("correct_no_snake", 2, "correct no-snake prediction"),
        ("failure", 2, "model failure case"),
    ):
        for sample in groups[key][:needed]:
            sample["selection_reason"] = reason
            manifest.append(sample)

    if len(manifest) < 6:
        fallback_pool = groups["failure"][2:] + groups["correct_snake"][2:] + groups["correct_no_snake"][2:]
        fallback_pool.sort(key=lambda item: item["confidence"])
        for sample in fallback_pool:
            if len(manifest) >= 6:
                break
            sample["selection_reason"] = "low-confidence fallback example"
            manifest.append(sample)

    return manifest[:6]
