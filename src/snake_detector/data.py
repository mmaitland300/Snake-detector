from __future__ import annotations

import csv
import random
import shutil
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Suffixes accepted on disk for raw/split datasets (aligned with collect download validation).
RAW_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".mpo"}
)


@dataclass(slots=True)
class SplitResult:
    train_count: int
    val_count: int
    test_count: int


def preprocess_pixels_inception(image: np.ndarray) -> np.ndarray:
    """Normalize [0,255] RGB image values to InceptionV3 range [-1,1]."""
    image = image.astype("float32")
    return (image / 127.5) - 1.0


def split_dataset(
    raw_dir: Path,
    split_dir: Path,
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42,
    manifest_path: Path | None = None,
    group_by: str = "observation_id",
) -> SplitResult:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")
    if train_split <= 0 or train_split >= 1:
        raise ValueError("train_split must be between 0 and 1.")
    if val_split <= 0 or val_split >= 1:
        raise ValueError("val_split must be between 0 and 1.")

    image_paths = [
        str(p) for p in raw_dir.rglob("*") if p.suffix.lower() in RAW_IMAGE_EXTENSIONS
    ]
    if not image_paths:
        raise FileNotFoundError(f"No images found under {raw_dir}")

    if manifest_path is not None:
        train_paths, val_paths, test_paths = _grouped_split_paths(
            image_paths=image_paths,
            raw_dir=raw_dir,
            manifest_path=manifest_path,
            group_by=group_by,
            train_split=train_split,
            val_split=val_split,
            seed=seed,
        )
    else:
        random.seed(seed)
        random.shuffle(image_paths)

        train_cut = int(len(image_paths) * train_split)
        train_paths = image_paths[:train_cut]
        test_paths = image_paths[train_cut:]

        val_cut = int(len(train_paths) * val_split)
        val_paths = train_paths[:val_cut]
        train_paths = train_paths[val_cut:]

    _clear_split_output(split_dir)

    labels = sorted({Path(p).parent.name for p in image_paths})
    _ensure_split_label_dirs(split_dir, labels)

    datasets = {
        "training": train_paths,
        "validation": val_paths,
        "testing": test_paths,
    }

    for split_name, split_paths in datasets.items():
        for src in split_paths:
            src_path = Path(src)
            label = src_path.parent.name
            dst_dir = split_dir / split_name / label
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_dir / src_path.name)

    return SplitResult(
        train_count=len(train_paths),
        val_count=len(val_paths),
        test_count=len(test_paths),
    )


def _clear_split_output(split_dir: Path) -> None:
    for split_name in ("training", "validation", "testing"):
        split_path = split_dir / split_name
        if split_path.exists():
            shutil.rmtree(split_path)


def _ensure_split_label_dirs(split_dir: Path, labels: Iterable[str]) -> None:
    """Create train/val/test class folders so downstream loaders see a stable layout."""
    for split_name in ("training", "validation", "testing"):
        for label in labels:
            (split_dir / split_name / label).mkdir(parents=True, exist_ok=True)


def count_split_images(split_root: Path) -> int:
    """Count image files under a split root (class subdirectories)."""
    if not split_root.is_dir():
        return 0
    return sum(
        1
        for path in split_root.rglob("*")
        if path.is_file() and path.suffix.lower() in RAW_IMAGE_EXTENSIONS
    )


def count_split_images_per_class(split_root: Path) -> dict[str, int]:
    """Count images per class folder (top-level subdirectories only)."""
    if not split_root.is_dir():
        return {}
    counts: dict[str, int] = {}
    for class_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        n = sum(
            1
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in RAW_IMAGE_EXTENSIONS
        )
        counts[class_dir.name] = n
    return counts


def keras_balanced_class_weights(training_dir: Path) -> dict[int, float] | None:
    """``class_weight`` for ``model.fit`` matching ``flow_from_directory`` class index order.

    Uses the sklearn balanced formula ``n_samples / (n_classes * count)`` per class.
    Returns ``None`` if fewer than two classes or any class has zero images.
    """
    counts_map = count_split_images_per_class(training_dir)
    if len(counts_map) < 2:
        return None
    class_names = sorted(counts_map.keys())
    counts = [counts_map[name] for name in class_names]
    if any(c <= 0 for c in counts):
        return None
    n_samples = float(sum(counts))
    n_classes = float(len(counts))
    return {idx: n_samples / (n_classes * float(counts[idx])) for idx in range(len(class_names))}


def _grouped_split_paths(
    *,
    image_paths: list[str],
    raw_dir: Path,
    manifest_path: Path,
    group_by: str,
    train_split: float,
    val_split: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    key_to_group = _load_group_mapping(manifest_path=manifest_path, group_by=group_by)
    grouped_by_label: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    missing: list[str] = []

    for src in image_paths:
        src_path = Path(src)
        key = (src_path.parent.name, src_path.stem)
        group = key_to_group.get(key)
        if not group:
            missing.append(str(src_path.relative_to(raw_dir)))
            continue
        grouped_by_label[src_path.parent.name][group].append(src)

    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} image(s) in {raw_dir} were not found in manifest {manifest_path}. "
            f"Examples: {sample}"
        )

    if not grouped_by_label:
        raise ValueError(f"No grouped records found using `{group_by}` from manifest {manifest_path}")

    rng = random.Random(seed)
    train_paths: list[str] = []
    val_paths: list[str] = []
    test_paths: list[str] = []
    for label in sorted(grouped_by_label.keys()):
        label_groups = grouped_by_label[label]
        group_ids = list(label_groups.keys())
        rng.shuffle(group_ids)

        label_train_groups, label_val_groups, label_test_groups = _allocate_label_group_ids(
            group_ids, train_split=train_split, val_split=val_split
        )

        train_paths.extend(src for group in label_train_groups for src in label_groups[group])
        val_paths.extend(src for group in label_val_groups for src in label_groups[group])
        test_paths.extend(src for group in label_test_groups for src in label_groups[group])
    return train_paths, val_paths, test_paths


def _load_group_mapping(*, manifest_path: Path, group_by: str) -> dict[tuple[str, str], str]:
    mapping: dict[tuple[str, str], str] = {}
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {manifest_path}")
        required = {"label", "provider", "image_id", group_by}
        missing_fields = sorted(required.difference(reader.fieldnames))
        if missing_fields:
            raise ValueError(
                f"Manifest missing required field(s) for grouped split: {', '.join(missing_fields)}"
            )

        for row_idx, row in enumerate(reader, start=2):
            label = (row.get("label") or "").strip()
            provider = (row.get("provider") or "").strip()
            image_id = (row.get("image_id") or "").strip()
            group_value = (row.get(group_by) or "").strip()
            if not label or not provider or not image_id or not group_value:
                continue
            key = (label, f"{provider}_{image_id}")
            previous = mapping.get(key)
            if previous is not None and previous != group_value:
                raise ValueError(
                    f"Conflicting group value for {key} in manifest {manifest_path} at row {row_idx}: "
                    f"{previous!r} vs {group_value!r}"
                )
            mapping.setdefault(key, group_value)
    return mapping


def _allocate_label_group_ids(
    shuffled_group_ids: list[str],
    *,
    train_split: float,
    val_split: float,
) -> tuple[list[str], list[str], list[str]]:
    """Assign groups to train/val/test with small-count safeguards.

    - ``n == 1``: all groups go to train (no val/test for that label).
    - ``n >= 2``: at least one group in train and one in test.
    - After reserving test groups, if only one group remains it all goes to train (no validation).
      If two remain, they split one train / one validation. Larger pools use ``val_split`` with at least
      one validation group when possible without emptying train.
    """
    n = len(shuffled_group_ids)
    if n == 0:
        return [], [], []
    if n == 1:
        return list(shuffled_group_ids), [], []

    test_count = max(1, min(n - 1, int(round(n * (1.0 - train_split)))))
    remaining = n - test_count
    assert remaining >= 1

    if remaining == 1:
        val_count = 0
    elif remaining == 2:
        val_count = 1
    else:
        val_count = max(1, int(round(remaining * val_split)))
        if val_count >= remaining:
            val_count = remaining - 1

    train_count = remaining - val_count
    if train_count < 1:
        val_count = remaining - 1
        train_count = 1

    test_groups = shuffled_group_ids[:test_count]
    val_groups = shuffled_group_ids[test_count : test_count + val_count]
    train_groups = shuffled_group_ids[test_count + val_count :]
    assert len(train_groups) + len(val_groups) + len(test_groups) == n
    assert len(train_groups) >= 1 and len(test_groups) >= 1
    return train_groups, val_groups, test_groups


def build_generators(
    split_dir: Path, image_size: int, batch_size: int, seed: int = 42
) -> tuple[Any, Any | None, Any | None]:
    """Build tf.keras generators with consistent Inception preprocessing.

    When a split has no images (e.g. grouped split with one observation per label),
    the corresponding generator is ``None`` so callers can skip validation or test
    evaluation instead of failing inside ``flow_from_directory``.
    """
    try:
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for data generators. Install with `pip install .[ml]`."
        ) from exc

    train_aug = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    eval_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

    common_args = {
        "class_mode": "binary",
        "target_size": (image_size, image_size),
        "color_mode": "rgb",
        "batch_size": batch_size,
    }
    train_gen = train_aug.flow_from_directory(
        str(split_dir / "training"),
        shuffle=True,
        seed=seed,
        **common_args,
    )
    val_gen = None
    if count_split_images(split_dir / "validation") > 0:
        val_gen = eval_aug.flow_from_directory(
            str(split_dir / "validation"),
            shuffle=False,
            **common_args,
        )
    test_gen = None
    if count_split_images(split_dir / "testing") > 0:
        test_gen = eval_aug.flow_from_directory(
            str(split_dir / "testing"),
            shuffle=False,
            **common_args,
        )
    return train_gen, val_gen, test_gen
