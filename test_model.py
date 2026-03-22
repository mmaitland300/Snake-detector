"""Legacy entrypoint retained for compatibility.

Use `python -m snake_detector.cli eval` instead.
Original script archived at `legacy/test_model_legacy.py`.
"""

from snake_detector.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["eval"]))
