"""Legacy entrypoint retained for compatibility.

Use `python -m snake_detector.cli train` instead.
Original script archived at `legacy/train_model_legacy.py`.
"""

from snake_detector.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train"]))
