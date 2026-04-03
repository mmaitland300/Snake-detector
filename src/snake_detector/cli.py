from __future__ import annotations

import argparse
import json
from pathlib import Path

from .collect import (
    DEFAULT_USER_AGENT,
    collect_inaturalist_records,
    download_images_from_manifest,
    write_manifest,
)
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
    split_cmd.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional manifest CSV used for grouped splitting.",
    )
    split_cmd.add_argument(
        "--group-by",
        type=str,
        default="observation_id",
        help="Manifest column used to group rows for split assignment.",
    )

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
    train_cmd.add_argument("--confusion-matrix-path", type=Path, default=None)
    train_cmd.add_argument("--predictions-panel-path", type=Path, default=None)
    train_cmd.add_argument("--predictions-manifest-path", type=Path, default=None)

    eval_cmd = subparsers.add_parser("eval", help="Evaluate a saved model.")
    eval_cmd.add_argument("--split-dir", type=Path, default=Path("data/split"))
    eval_cmd.add_argument("--model-path", type=Path, default=Path("artifacts/model.keras"))
    eval_cmd.add_argument("--metrics-path", type=Path, default=Path("artifacts/metrics.json"))
    eval_cmd.add_argument("--image-size", type=int, default=150)
    eval_cmd.add_argument("--batch-size", type=int, default=20)
    eval_cmd.add_argument("--backbone", type=str, default="inceptionv3")
    eval_cmd.add_argument("--confusion-matrix-path", type=Path, default=None)
    eval_cmd.add_argument("--predictions-panel-path", type=Path, default=None)
    eval_cmd.add_argument("--predictions-manifest-path", type=Path, default=None)

    predict_cmd = subparsers.add_parser("predict", help="Predict on one image.")
    predict_cmd.add_argument("--model-path", type=Path, default=Path("artifacts/model.keras"))
    predict_cmd.add_argument("--image-path", type=Path, required=True)
    predict_cmd.add_argument("--image-size", type=int, default=150)

    collect_cmd = subparsers.add_parser(
        "collect-inat",
        help="Build a real-image manifest from iNaturalist observation photos.",
    )
    collect_cmd.add_argument("--label", type=str, required=True)
    collect_cmd.add_argument("--manifest-path", type=Path, default=Path("data/manifests/inat.csv"))
    collect_cmd.add_argument(
        "--taxon-name",
        type=str,
        default=None,
        help="Scientific name; resolved via /v1/taxa then queried by taxon_id (see docs/real_image_collection.md).",
    )
    collect_cmd.add_argument(
        "--taxon-id",
        type=int,
        default=None,
        help="Skip taxa search and query observations by this iNaturalist taxon id.",
    )
    collect_cmd.add_argument("--max-images", type=int, default=200)
    collect_cmd.add_argument(
        "--per-page",
        type=int,
        default=50,
        metavar="N",
        help="Observations API page size (1-200). Smaller pages can reduce CDN 403s on large pulls.",
    )
    collect_cmd.add_argument("--quality-grade", type=str, default="research")
    collect_cmd.add_argument("--photo-size", type=str, default="large")
    collect_cmd.add_argument("--captive", type=str, default="false")
    collect_cmd.add_argument(
        "--license",
        dest="licenses",
        action="append",
        default=None,
        help="Allowed photo license code. Repeat flag to allow multiple values.",
    )
    collect_cmd.add_argument("--append", action="store_true")
    collect_cmd.add_argument(
        "--start-page",
        type=int,
        default=1,
        metavar="N",
        help="First observations API page to fetch (1-based). Use with --max-pages for chunked collection.",
    )
    collect_cmd.add_argument(
        "--max-pages",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of observation pages to fetch this run (omit for no limit).",
    )
    collect_cmd.add_argument(
        "--flush-every-page",
        action="store_true",
        help="Append manifest after each API page (checkpointing). Requires a stable --manifest-path.",
    )
    collect_cmd.add_argument("--user-agent", type=str, default=DEFAULT_USER_AGENT)

    download_cmd = subparsers.add_parser(
        "download-manifest",
        help="Download images referenced by a CSV manifest into raw class folders.",
    )
    download_cmd.add_argument("--manifest-path", type=Path, required=True)
    download_cmd.add_argument("--output-dir", type=Path, default=Path("data/raw_real"))
    download_cmd.add_argument("--overwrite", action="store_true")
    download_cmd.add_argument("--user-agent", type=str, default=DEFAULT_USER_AGENT)

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
            manifest_path=args.manifest_path,
            group_by=args.group_by,
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
        if args.confusion_matrix_path is not None:
            cfg.paths.confusion_matrix_path = args.confusion_matrix_path
        if args.predictions_panel_path is not None:
            cfg.paths.predictions_panel_path = args.predictions_panel_path
        if args.predictions_manifest_path is not None:
            cfg.paths.predictions_manifest_path = args.predictions_manifest_path
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
        if args.confusion_matrix_path is not None:
            cfg.paths.confusion_matrix_path = args.confusion_matrix_path
        if args.predictions_panel_path is not None:
            cfg.paths.predictions_panel_path = args.predictions_panel_path
        if args.predictions_manifest_path is not None:
            cfg.paths.predictions_manifest_path = args.predictions_manifest_path
        metrics = run_evaluate(cfg)
        print(json.dumps(metrics, indent=2))
        return 0

    if args.command == "predict":
        label, confidence = run_predict(args.model_path, args.image_path, args.image_size)
        print(json.dumps({"label": label, "confidence": confidence}, indent=2))
        return 0

    if args.command == "collect-inat":
        if not 1 <= args.per_page <= 200:
            parser.error("--per-page must be between 1 and 200")
        if args.start_page < 1:
            parser.error("--start-page must be >= 1")
        if args.max_pages is not None and args.max_pages < 1:
            parser.error("--max-pages must be >= 1 when provided")
        if args.flush_every_page and args.manifest_path.exists() and not args.append:
            parser.error(
                "Refusing to overwrite existing manifest: use --append with --flush-every-page when resuming, "
                "or choose a new --manifest-path."
            )
        result = collect_inaturalist_records(
            label=args.label,
            max_images=args.max_images,
            taxon_name=args.taxon_name,
            taxon_id=args.taxon_id,
            quality_grade=args.quality_grade,
            per_page=args.per_page,
            photo_size=args.photo_size,
            captive=args.captive,
            photo_license_allowlist=tuple(args.licenses or ("CC0", "CC-BY", "CC-BY-SA")),
            user_agent=args.user_agent,
            start_page=args.start_page,
            max_pages=args.max_pages,
            manifest_path=args.manifest_path if args.flush_every_page else None,
            flush_every_page=args.flush_every_page,
        )
        if args.flush_every_page:
            rows_written = result.manifest_rows_written
            rows_skipped = result.manifest_rows_skipped_duplicate
        else:
            rows_written, rows_skipped = write_manifest(
                result.records,
                args.manifest_path,
                append=args.append,
                dedupe=args.append,
            )
        print(
            json.dumps(
                {
                    "manifest_path": str(args.manifest_path),
                    "records_collected": len(result.records),
                    "records_written": rows_written,
                    "records_skipped_duplicate": rows_skipped,
                    "label": args.label,
                    "start_page": result.start_page,
                    "next_page": result.next_page,
                    "pages_fetched": result.pages_fetched,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "download-manifest":
        result = download_images_from_manifest(
            manifest_path=args.manifest_path,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            user_agent=args.user_agent,
        )
        result["output_dir"] = str(args.output_dir)
        print(json.dumps(result, indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
