from snake_detector.cli import build_parser


def test_cli_has_expected_subcommands() -> None:
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices.keys()  # noqa: SLF001
    assert {"split", "train", "eval", "predict"}.issubset(set(commands))


def test_predict_requires_image_path() -> None:
    parser = build_parser()
    args = parser.parse_args(["predict", "--image-path", "example.jpg"])
    assert args.command == "predict"


def test_train_accepts_backbone_override() -> None:
    parser = build_parser()
    args = parser.parse_args(["train", "--backbone", "sklearn_mlp"])
    assert args.backbone == "sklearn_mlp"
