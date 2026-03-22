from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
) -> SplitResult:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")
    if train_split <= 0 or train_split >= 1:
        raise ValueError("train_split must be between 0 and 1.")
    if val_split <= 0 or val_split >= 1:
        raise ValueError("val_split must be between 0 and 1.")

    image_paths = [str(p) for p in raw_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    random.seed(seed)
    random.shuffle(image_paths)

    train_cut = int(len(image_paths) * train_split)
    train_paths = image_paths[:train_cut]
    test_paths = image_paths[train_cut:]

    val_cut = int(len(train_paths) * val_split)
    val_paths = train_paths[:val_cut]
    train_paths = train_paths[val_cut:]

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


def build_generators(split_dir: Path, image_size: int, batch_size: int):
    """Build tf.keras generators with consistent Inception preprocessing."""
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
        **common_args,
    )
    val_gen = eval_aug.flow_from_directory(
        str(split_dir / "validation"),
        shuffle=False,
        **common_args,
    )
    test_gen = eval_aug.flow_from_directory(
        str(split_dir / "testing"),
        shuffle=False,
        **common_args,
    )
    return train_gen, val_gen, test_gen
