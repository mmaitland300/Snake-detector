import pytest

from snake_detector.cli import build_parser
from snake_detector.cli import main as cli_main


def test_cli_has_expected_subcommands() -> None:
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices.keys()  # noqa: SLF001
    assert {"split", "train", "eval", "predict", "collect-inat", "download-manifest"}.issubset(
        set(commands)
    )


def test_predict_requires_image_path() -> None:
    parser = build_parser()
    args = parser.parse_args(["predict", "--image-path", "example.jpg"])
    assert args.command == "predict"


def test_train_accepts_backbone_override() -> None:
    parser = build_parser()
    args = parser.parse_args(["train", "--backbone", "sklearn_mlp"])
    assert args.backbone == "sklearn_mlp"


def test_collect_inat_accepts_taxon_name() -> None:
    parser = build_parser()
    args = parser.parse_args(["collect-inat", "--label", "snake", "--taxon-name", "Serpentes"])
    assert args.command == "collect-inat"
    assert args.taxon_name == "Serpentes"
    assert args.per_page == 50
    assert args.start_page == 1
    assert args.max_pages is None
    assert args.flush_every_page is False


def test_collect_inat_accepts_resume_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "collect-inat",
            "--label",
            "snake",
            "--taxon-name",
            "Serpentes",
            "--start-page",
            "21",
            "--max-pages",
            "15",
            "--flush-every-page",
            "--append",
        ]
    )
    assert args.start_page == 21
    assert args.max_pages == 15
    assert args.flush_every_page is True
    assert args.append is True


def test_collect_inat_rejects_flush_over_existing_without_append(tmp_path) -> None:
    manifest = tmp_path / "existing.csv"
    manifest.write_bytes(b"x")
    with pytest.raises(SystemExit):
        cli_main(
            [
                "collect-inat",
                "--label",
                "snake",
                "--taxon-name",
                "Serpentes",
                "--manifest-path",
                str(manifest),
                "--max-images",
                "1",
                "--flush-every-page",
            ]
        )


def test_split_accepts_manifest_grouping_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "split",
            "--raw-dir",
            "data/raw",
            "--split-dir",
            "data/split",
            "--manifest-path",
            "data/manifests/pilot_real.csv",
            "--group-by",
            "observation_id",
        ]
    )
    assert args.command == "split"
    assert str(args.manifest_path).endswith("pilot_real.csv")
    assert args.group_by == "observation_id"


def test_train_accepts_artifact_output_path_overrides() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--confusion-matrix-path",
            "artifacts/pilot_v2/confusion.png",
            "--predictions-panel-path",
            "artifacts/pilot_v2/panel.png",
            "--predictions-manifest-path",
            "artifacts/pilot_v2/preds.json",
        ]
    )
    assert args.confusion_matrix_path.parts[-2:] == ("pilot_v2", "confusion.png")
    assert args.predictions_panel_path.parts[-2:] == ("pilot_v2", "panel.png")
    assert args.predictions_manifest_path.parts[-2:] == ("pilot_v2", "preds.json")
