"""Ensure `artifacts/model.joblib` exists and passes a minimum size check (idempotent across restarts)."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_URL = (
    "https://github.com/mmaitland300/Snake-detector/releases/download/v1.0.0/model.joblib"
)
MODEL_PATH = REPO_ROOT / "artifacts" / "model.joblib"
_DOWNLOAD_TIMEOUT_SEC = 120
# Also enforced in `app/gradio_app._MIN_ARTIFACT_BYTES` when resolving predict paths.
_MIN_BYTES = 512


def _ensure_model() -> None:
    """Skip if artifact exists and meets minimum size; else remove undersized junk and download."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.is_file():
        size = MODEL_PATH.stat().st_size
        if size >= _MIN_BYTES:
            return
        MODEL_PATH.unlink(missing_ok=True)

    req = urllib.request.Request(
        MODEL_URL,
        headers={"User-Agent": "Snake-detector-HF-Space/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT_SEC) as resp:
            status = getattr(resp, "status", None)
            if status is not None and status != 200:
                raise RuntimeError(
                    f"Model download returned HTTP {status}. "
                    "Upload artifacts/model.joblib in the Space Files tab as a fallback."
                )
            data = resp.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not download model from {MODEL_URL} (timeout {_DOWNLOAD_TIMEOUT_SEC}s). "
            "Upload artifacts/model.joblib in the Space Files tab as a fallback."
        ) from exc

    if len(data) < _MIN_BYTES:
        raise RuntimeError(
            "Downloaded file is too small to be a valid model artifact. "
            "Upload artifacts/model.joblib in the Space Files tab as a fallback."
        )

    tmp_path = MODEL_PATH.parent / f".model-download-{os.getpid()}.tmp"
    try:
        tmp_path.write_bytes(data)
        tmp_path.replace(MODEL_PATH)
    finally:
        tmp_path.unlink(missing_ok=True)
