"""Hugging Face Spaces default entry: re-exports `demo` from `app/gradio_app.py` (no `launch()` at import)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "snake_detector_gradio_app",
    _ROOT / "app" / "gradio_app.py",
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Cannot load app/gradio_app.py")
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
demo = _MOD.demo

if __name__ == "__main__":
    demo.launch()
