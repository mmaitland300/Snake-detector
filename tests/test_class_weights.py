"""Balanced class weights for Keras training."""

from __future__ import annotations

from pathlib import Path

from snake_detector.cli import build_parser
from snake_detector.data import keras_balanced_class_weights


def test_keras_balanced_class_weights_matches_flow_order(tmp_path: Path) -> None:
    train = tmp_path / "training"
    (train / "no_snake").mkdir(parents=True)
    (train / "snake").mkdir(parents=True)
    for i in range(9):
        (train / "no_snake" / f"n{i}.jpg").write_bytes(b"")
    (train / "snake" / "s0.jpg").write_bytes(b"")

    cw = keras_balanced_class_weights(train)
    assert cw is not None
    assert cw.keys() == {0, 1}
    # sklearn balanced: n / (n_classes * count); minority class gets larger weight
    assert cw[1] > cw[0]


def test_keras_balanced_class_weights_returns_none_for_single_class(tmp_path: Path) -> None:
    train = tmp_path / "training"
    (train / "only").mkdir(parents=True)
    (train / "only" / "a.jpg").write_bytes(b"")
    assert keras_balanced_class_weights(train) is None


def test_train_cli_no_class_weights_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["train", "--no-class-weights"])
    assert args.no_class_weights is True
