from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import AppConfig
from .data import split_dataset
from .evaluate import run_evaluate
from .predict import run_predict
from .train import run_train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="snake-detector")
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_cmd = subparsers.add_parser("split", help="Split raw dataset into train/val/test folders.")
    split_cmd.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    split_cmd.add_argument("--split-dir", type=Path, default=Path("data/split"))
    split_cmd.add_argument("--train-split", type=float, default=0.8)
    split_cmd.add_argument("--val-split", type=float, default=0.1)
    split_cmd.add_argument("--seed", type=int, default=42)

    train_cmd = subparsers.add_parser("train", help="Train model and save artifacts.")
    train_cmd.add_argument("--split-dir", type=Path, default=Path("data/split"))
    train_cmd.add_argument("--model-path", type=Path, default=Path("artifacts/model.keras"))
    train_cmd.add_argument("--metrics-path", type=Path, default=Path("artifacts/metrics.json"))
    train_cmd.add_argument("--image-size", type=int, default=150)
    train_cmd.add_argument("--batch-size", type=int, default=20)
    train_cmd.add_argument("--epochs", type=int, default=10)
    train_cmd.add_argument("--learning-rate", type=float, default=1e-4)
    train_cmd.add_argument("--seed", type=int, default=42)
    train_cmd.add_argument("--backbone", type=str, default="inceptionv3")

    eval_cmd = subparsers.add_parser("eval", help="Evaluate a saved model.")
    eval_cmd.add_argument("--split-dir", type=Path, default=Path("data/split"))
    eval_cmd.add_argument("--model-path", type=Path, default=Path("artifacts/model.keras"))
    eval_cmd.add_argument("--metrics-path", type=Path, default=Path("artifacts/metrics.json"))
    eval_cmd.add_argument("--image-size", type=int, default=150)
    eval_cmd.add_argument("--batch-size", type=int, default=20)
    eval_cmd.add_argument("--backbone", type=str, default="inceptionv3")

    predict_cmd = subparsers.add_parser("predict", help="Predict on one image.")
    predict_cmd.add_argument("--model-path", type=Path, default=Path("artifacts/model.keras"))
    predict_cmd.add_argument("--image-path", type=Path, required=True)
    predict_cmd.add_argument("--image-size", type=int, default=150)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "split":
        result = split_dataset(
            raw_dir=args.raw_dir,
            split_dir=args.split_dir,
            train_split=args.train_split,
            val_split=args.val_split,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "train_count": result.train_count,
                    "val_count": result.val_count,
                    "test_count": result.test_count,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "train":
        cfg = AppConfig()
        cfg.data.split_dir = args.split_dir
        cfg.data.seed = args.seed
        cfg.paths.model_path = args.model_path
        cfg.paths.metrics_path = args.metrics_path
        cfg.train.image_size = args.image_size
        cfg.train.batch_size = args.batch_size
        cfg.train.epochs = args.epochs
        cfg.train.learning_rate = args.learning_rate
        cfg.train.backbone = args.backbone
        metrics = run_train(cfg)
        print(json.dumps(metrics, indent=2))
        return 0

    if args.command == "eval":
        cfg = AppConfig()
        cfg.data.split_dir = args.split_dir
        cfg.paths.model_path = args.model_path
        cfg.paths.metrics_path = args.metrics_path
        cfg.train.image_size = args.image_size
        cfg.train.batch_size = args.batch_size
        cfg.train.backbone = args.backbone
        metrics = run_evaluate(cfg)
        print(json.dumps(metrics, indent=2))
        return 0

    if args.command == "predict":
        label, confidence = run_predict(args.model_path, args.image_path, args.image_size)
        print(json.dumps({"label": label, "confidence": confidence}, indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
