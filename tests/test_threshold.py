"""Decision threshold resolution for eval/predict parity with deployment."""

from __future__ import annotations

import pytest

from snake_detector.pipeline import resolve_decision_threshold


def test_resolve_decision_threshold_explicit() -> None:
    assert resolve_decision_threshold(0.72) == 0.72


def test_resolve_decision_threshold_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        resolve_decision_threshold(1.5)
    with pytest.raises(ValueError):
        resolve_decision_threshold(-0.01)


def test_resolve_decision_threshold_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SNAKE_DETECTOR_THRESHOLD", "0.33")
    assert resolve_decision_threshold(None) == 0.33


def test_resolve_decision_threshold_env_ignored_when_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SNAKE_DETECTOR_THRESHOLD", "0.1")
    assert resolve_decision_threshold(0.9) == 0.9


def test_resolve_decision_threshold_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SNAKE_DETECTOR_THRESHOLD", raising=False)
    assert resolve_decision_threshold(None) == 0.5


def test_resolve_decision_threshold_env_invalid_string_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SNAKE_DETECTOR_THRESHOLD", "not-a-float")
    with pytest.raises(ValueError, match="SNAKE_DETECTOR_THRESHOLD"):
        resolve_decision_threshold(None)


def test_resolve_decision_threshold_env_out_of_range_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SNAKE_DETECTOR_THRESHOLD", "2.0")
    with pytest.raises(ValueError, match="SNAKE_DETECTOR_THRESHOLD"):
        resolve_decision_threshold(None)
