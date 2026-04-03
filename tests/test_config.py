from pathlib import Path

from snake_detector.config import AppConfig, save_config


def test_app_config_defaults() -> None:
    cfg = AppConfig()
    assert cfg.data.train_split == 0.8
    assert cfg.train.backbone == "inceptionv3"
    assert cfg.train.use_class_weights is True
    assert cfg.inference.decision_threshold == 0.5
    assert cfg.paths.model_path.name == "model.keras"


def test_save_config_writes_json(tmp_path: Path) -> None:
    cfg = AppConfig()
    output = tmp_path / "run_config.json"
    save_config(cfg, output)
    assert output.exists()
    content = output.read_text(encoding="utf-8")
    assert '"backbone": "inceptionv3"' in content
    assert '"use_class_weights": true' in content
    assert '"decision_threshold": 0.5' in content
