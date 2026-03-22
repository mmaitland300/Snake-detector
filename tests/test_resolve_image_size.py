import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from snake_detector.pipeline import resolve_prediction_image_size


def test_resolve_prediction_image_size_joblib_uses_run_config(tmp_path: Path) -> None:
    clf = DummyClassifier(strategy="uniform")
    clf.fit(np.zeros((4, 8)), np.array([0, 1, 0, 1]))
    artifact = {
        "backend": "sklearn_mlp",
        "feature_size": 48,
        "class_names": ["no_snake", "snake"],
        "classifier": clf,
    }
    model_path = tmp_path / "model.joblib"
    joblib.dump(artifact, model_path)
    run_cfg = tmp_path / "run_config.json"
    run_cfg.write_text(json.dumps({"train": {"image_size": 128}}), encoding="utf-8")

    assert resolve_prediction_image_size(model_path) == 128


def test_resolve_prediction_image_size_joblib_defaults_without_run_config(tmp_path: Path) -> None:
    clf = DummyClassifier(strategy="uniform")
    clf.fit(np.zeros((4, 8)), np.array([0, 1, 0, 1]))
    artifact = {
        "backend": "sklearn_mlp",
        "feature_size": 48,
        "class_names": ["no_snake", "snake"],
        "classifier": clf,
    }
    model_path = tmp_path / "model.joblib"
    joblib.dump(artifact, model_path)

    assert resolve_prediction_image_size(model_path, default=150) == 150


@pytest.mark.parametrize(
    "env_value,expected",
    [
        ("64", 64),
        ("  200  ", 200),
    ],
)
def test_resolve_prediction_image_size_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env_value: str, expected: int
) -> None:
    monkeypatch.setenv("SNAKE_DETECTOR_IMAGE_SIZE", env_value)
    clf = DummyClassifier(strategy="uniform")
    clf.fit(np.zeros((4, 8)), np.array([0, 1, 0, 1]))
    artifact = {
        "backend": "sklearn_mlp",
        "feature_size": 48,
        "class_names": ["no_snake", "snake"],
        "classifier": clf,
    }
    model_path = tmp_path / "model.joblib"
    joblib.dump(artifact, model_path)
    (tmp_path / "run_config.json").write_text(
        json.dumps({"train": {"image_size": 1}}), encoding="utf-8"
    )

    assert resolve_prediction_image_size(model_path) == expected
