from __future__ import annotations

import csv
from pathlib import Path

import pytest

from snake_detector.data import count_split_images, split_dataset


def _touch_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-a-real-image")


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["label", "provider", "image_id", "observation_id", "image_url"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_grouped_split_keeps_observation_in_single_split(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    split_dir = tmp_path / "split"
    manifest_path = tmp_path / "manifest.csv"

    # Two images per observation; grouped split must keep each observation together.
    rows: list[dict[str, str]] = []
    for idx, observation_id in enumerate(("obs_a", "obs_a", "obs_b", "obs_b"), start=1):
        image_id = str(idx)
        _touch_image(raw_dir / "snake" / f"inaturalist_{image_id}.jpg")
        rows.append(
            {
                "label": "snake",
                "provider": "inaturalist",
                "image_id": image_id,
                "observation_id": observation_id,
                "image_url": f"https://example/{image_id}.jpg",
            }
        )
    _write_manifest(manifest_path, rows)

    result = split_dataset(
        raw_dir=raw_dir,
        split_dir=split_dir,
        train_split=0.8,
        val_split=0.1,
        seed=42,
        manifest_path=manifest_path,
        group_by="observation_id",
    )

    assert result.train_count + result.val_count + result.test_count == 4
    split_by_observation: dict[str, set[str]] = {"obs_a": set(), "obs_b": set()}
    stem_to_observation = {f"inaturalist_{row['image_id']}": row["observation_id"] for row in rows}

    for split_name in ("training", "validation", "testing"):
        for image_path in (split_dir / split_name).rglob("*.jpg"):
            obs = stem_to_observation[image_path.stem]
            split_by_observation[obs].add(split_name)

    assert all(len(splits) == 1 for splits in split_by_observation.values())


def test_grouped_split_fails_when_raw_image_missing_from_manifest(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    split_dir = tmp_path / "split"
    manifest_path = tmp_path / "manifest.csv"

    _touch_image(raw_dir / "snake" / "inaturalist_1.jpg")
    _touch_image(raw_dir / "snake" / "inaturalist_2.jpg")
    _write_manifest(
        manifest_path,
        [
            {
                "label": "snake",
                "provider": "inaturalist",
                "image_id": "1",
                "observation_id": "obs_1",
                "image_url": "https://example/1.jpg",
            }
        ],
    )

    with pytest.raises(ValueError, match="were not found in manifest"):
        split_dataset(
            raw_dir=raw_dir,
            split_dir=split_dir,
            train_split=0.8,
            val_split=0.1,
            seed=42,
            manifest_path=manifest_path,
            group_by="observation_id",
        )


def test_split_rerun_clears_stale_outputs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    split_dir = tmp_path / "split"

    keep_path = raw_dir / "snake" / "inaturalist_1.jpg"
    stale_path = raw_dir / "snake" / "inaturalist_2.jpg"
    other_path = raw_dir / "no_snake" / "inaturalist_3.jpg"
    _touch_image(keep_path)
    _touch_image(stale_path)
    _touch_image(other_path)

    first = split_dataset(raw_dir=raw_dir, split_dir=split_dir, seed=7)
    assert first.train_count + first.val_count + first.test_count == 3

    stale_path.unlink()
    second = split_dataset(raw_dir=raw_dir, split_dir=split_dir, seed=7)
    assert second.train_count + second.val_count + second.test_count == 2

    # No stale file from the first run should remain in split output.
    assert not any(path.stem == "inaturalist_2" for path in split_dir.rglob("*.jpg"))
    assert sum(1 for _ in split_dir.rglob("*.jpg")) == 2


def test_grouped_split_is_stratified_by_label(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    split_dir = tmp_path / "split"
    manifest_path = tmp_path / "manifest.csv"

    rows: list[dict[str, str]] = []
    image_id = 1
    for label in ("snake", "no_snake"):
        for obs_idx in (1, 2, 3):
            for _ in range(2):
                _touch_image(raw_dir / label / f"inaturalist_{image_id}.jpg")
                rows.append(
                    {
                        "label": label,
                        "provider": "inaturalist",
                        "image_id": str(image_id),
                        "observation_id": f"{label}_obs_{obs_idx}",
                        "image_url": f"https://example/{image_id}.jpg",
                    }
                )
                image_id += 1
    _write_manifest(manifest_path, rows)

    split_dataset(
        raw_dir=raw_dir,
        split_dir=split_dir,
        train_split=0.8,
        val_split=0.1,
        seed=42,
        manifest_path=manifest_path,
        group_by="observation_id",
    )

    train_labels = {path.name for path in (split_dir / "training").iterdir() if path.is_dir()}
    test_labels = {path.name for path in (split_dir / "testing").iterdir() if path.is_dir()}
    assert train_labels == {"snake", "no_snake"}
    assert test_labels == {"snake", "no_snake"}


def test_grouped_split_raises_on_conflicting_manifest_group_mapping(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    split_dir = tmp_path / "split"
    manifest_path = tmp_path / "manifest.csv"

    _touch_image(raw_dir / "snake" / "inaturalist_1.jpg")
    _write_manifest(
        manifest_path,
        [
            {
                "label": "snake",
                "provider": "inaturalist",
                "image_id": "1",
                "observation_id": "obs_a",
                "image_url": "https://example/1.jpg",
            },
            {
                "label": "snake",
                "provider": "inaturalist",
                "image_id": "1",
                "observation_id": "obs_b",
                "image_url": "https://example/1.jpg",
            },
        ],
    )

    with pytest.raises(ValueError, match="Conflicting group value"):
        split_dataset(
            raw_dir=raw_dir,
            split_dir=split_dir,
            train_split=0.8,
            val_split=0.1,
            seed=42,
            manifest_path=manifest_path,
            group_by="observation_id",
        )


def test_grouped_split_small_label_keeps_train_and_test_when_two_groups(tmp_path: Path) -> None:
    """With only two observation groups per label, train and test each get at least one group."""
    raw_dir = tmp_path / "raw"
    split_dir = tmp_path / "split"
    manifest_path = tmp_path / "manifest.csv"

    rows: list[dict[str, str]] = []
    image_id = 1
    for label in ("snake", "no_snake"):
        for obs in ("a", "b"):
            _touch_image(raw_dir / label / f"inaturalist_{image_id}.jpg")
            rows.append(
                {
                    "label": label,
                    "provider": "inaturalist",
                    "image_id": str(image_id),
                    "observation_id": f"{label}_{obs}",
                    "image_url": f"https://example/{image_id}.jpg",
                }
            )
            image_id += 1
    _write_manifest(manifest_path, rows)

    split_dataset(
        raw_dir=raw_dir,
        split_dir=split_dir,
        train_split=0.8,
        val_split=0.1,
        seed=99,
        manifest_path=manifest_path,
        group_by="observation_id",
    )

    for label in ("snake", "no_snake"):
        assert any((split_dir / "training" / label).iterdir())
        assert any((split_dir / "testing" / label).iterdir())


def test_grouped_split_single_group_label_goes_to_train_only(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    split_dir = tmp_path / "split"
    manifest_path = tmp_path / "manifest.csv"

    _touch_image(raw_dir / "snake" / "inaturalist_1.jpg")
    _touch_image(raw_dir / "no_snake" / "inaturalist_2.jpg")
    _write_manifest(
        manifest_path,
        [
            {
                "label": "snake",
                "provider": "inaturalist",
                "image_id": "1",
                "observation_id": "only_snake_obs",
                "image_url": "https://example/1.jpg",
            },
            {
                "label": "no_snake",
                "provider": "inaturalist",
                "image_id": "2",
                "observation_id": "only_neg_obs",
                "image_url": "https://example/2.jpg",
            },
        ],
    )

    split_dataset(
        raw_dir=raw_dir,
        split_dir=split_dir,
        train_split=0.8,
        val_split=0.1,
        seed=1,
        manifest_path=manifest_path,
        group_by="observation_id",
    )

    assert list((split_dir / "training" / "snake").glob("*"))
    assert list((split_dir / "training" / "no_snake").glob("*"))
    # Class folders still exist under val/test so Keras can target a stable layout;
    # grouped single-observation labels leave them empty of images.
    for split_name in ("validation", "testing"):
        for label in ("snake", "no_snake"):
            class_dir = split_dir / split_name / label
            assert class_dir.is_dir()
            assert not list(class_dir.glob("*"))
    assert count_split_images(split_dir / "validation") == 0
    assert count_split_images(split_dir / "testing") == 0


def test_split_includes_webp_when_present(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    split_dir = tmp_path / "split"
    _touch_image(raw_dir / "snake" / "inaturalist_1.webp")
    split_dataset(raw_dir=raw_dir, split_dir=split_dir, seed=1)
    copied = list(split_dir.rglob("*.webp"))
    assert copied


def test_joblib_evaluate_skips_when_testing_split_has_no_images(tmp_path: Path) -> None:
    """Joblib eval must not np.stack an empty test set (grouped split can leave testing/ empty)."""
    joblib = pytest.importorskip("joblib")
    pytest.importorskip("sklearn")

    from snake_detector.config import AppConfig, DataConfig, PathsConfig, TrainConfig
    from snake_detector.pipeline import evaluate_saved_model

    split_dir = tmp_path / "split"
    for label in ("snake", "no_snake"):
        (split_dir / "testing" / label).mkdir(parents=True, exist_ok=True)
    assert count_split_images(split_dir / "testing") == 0

    model_path = tmp_path / "model.joblib"
    joblib.dump(
        {
            "backend": "sklearn_mlp",
            "feature_size": 16,
            "class_names": ["no_snake", "snake"],
            "classifier": object(),
        },
        model_path,
    )

    cfg = AppConfig(
        data=DataConfig(split_dir=split_dir, seed=1),
        train=TrainConfig(image_size=32, backbone="sklearn_mlp"),
        paths=PathsConfig(
            model_path=model_path,
            metrics_path=tmp_path / "metrics.json",
            confusion_matrix_path=tmp_path / "cm.png",
            predictions_panel_path=tmp_path / "panel.png",
            predictions_manifest_path=tmp_path / "preds.json",
        ),
    )
    metrics = evaluate_saved_model(cfg)
    assert metrics.get("test_evaluation_skipped") is True
    assert metrics.get("accuracy") is None
