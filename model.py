"""Legacy entrypoint retained for compatibility.

Current model definitions live in `src/snake_detector/models.py`.
Original prototype archived at `legacy/model_legacy.py`.
"""

from snake_detector.models import ModelSpec, build_binary_classifier

__all__ = ["ModelSpec", "build_binary_classifier"]
