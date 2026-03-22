"""Legacy entrypoint retained for compatibility.

Use `python -m snake_detector.cli split` instead.
Original script archived at `legacy/build_dataset_legacy.py`.
"""

from snake_detector.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["split"]))
