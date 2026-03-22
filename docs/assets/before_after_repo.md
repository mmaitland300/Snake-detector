# Repository Before/After

## Before (legacy)
- Flat root-level scripts only.
- Hardcoded Colab paths.
- No package layout, tests, CI, or deployment docs.

## After (overhaul)
- `src/snake_detector/` package with importable modules.
- Unified CLI entrypoint (`split`, `train`, `eval`, `predict`).
- Dataset/legal and deployment decision docs.
- Test suite + lint + CI workflow.
- Legacy scripts archived under `legacy/`.
