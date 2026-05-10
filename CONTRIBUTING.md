# Contributing

Thanks for taking a look at Snake Detector. Focused contributions are welcome when they improve reproducibility, documentation, tests, demo reliability, dataset transparency, or model-evaluation clarity.

This project is intentionally bounded: it is a `snake` vs `no_snake` image-classification demo, not species identification software and not field-ready wildlife safety software.

## Good First Contributions

- Reproducible bug reports for the CLI, local Gradio app, or Hugging Face demo path
- Docs fixes that clarify setup, model release assets, dataset lanes, or limitations
- Tests for split/train/eval/predict behavior
- Small improvements to error messages and validation
- Model-card, dataset-card, or release-note corrections
- Platform notes for Windows, macOS, Linux, or Hugging Face Spaces

Please open an issue before proposing broad architecture rewrites, new model families, new dataset sources, or safety/field-use features.

## Dataset And Image Policy

- Do not commit raw third-party wildlife photos to git.
- Do not upload private images, location-sensitive images, or files with private metadata.
- For real-photo work, document source, license posture, collection method, and whether raw images can be redistributed.
- Keep generated placeholder data separate from real-photo model/evaluation artifacts.
- Preserve the distinction between the Python package version and model release tags such as `v1.1.0-real-dataset`.

## Local Setup

The package targets Python 3.10+ and CI runs on Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -e .[dev,demo]
```

On macOS or Linux, use:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .[dev,demo]
```

Run the local app:

```bash
.venv\Scripts\python app/gradio_app.py
```

## Verification

Before opening a pull request, run the blocking CI checks:

```bash
ruff check .
pytest
```

For workflow changes, also run the relevant CLI command locally. For example:

```bash
python -m snake_detector.cli split --help
python -m snake_detector.cli train --help
python -m snake_detector.cli eval --help
python -m snake_detector.cli predict --help
```

If your change touches the Gradio demo, include the manual launch command and what you checked in the pull request description.

## Pull Request Guidelines

- Keep changes focused and reviewable.
- Explain what changed, why it changed, and how you tested it.
- Do not weaken documented limits around species ID, field safety, or dataset redistribution.
- Do not mix generated artifacts, release packaging, and code refactors unless the connection is necessary.
- Do not commit secrets, Hugging Face tokens, private file paths, raw scraped images, or location-sensitive metadata.

## Issue Reports

For normal bugs, include:

- Operating system
- Python version
- Install command used
- CLI command or app workflow
- Expected behavior
- Actual behavior
- Traceback or logs when available

For model/dataset reports, include the model or release version, image source/license context, and whether any private metadata has been removed.

For security-sensitive reports, use [SECURITY.md](SECURITY.md) instead of opening a public issue.
