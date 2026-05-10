# Security Policy

Snake Detector is a demo project that runs local CLI workflows and a Gradio app. Security reports are welcome when they involve realistic risk around file handling, model/demo deployment, dependency behavior, or accidental exposure of private data.

## Supported Scope

Please report issues related to the current `main` branch, the current Hugging Face Space deployment, and published GitHub Release assets.

## Please Report Privately

Do not open a public issue for vulnerabilities or reports that include private files, secrets, or location-sensitive image metadata.

Examples worth reporting privately:

- Unsafe file handling or path traversal in CLI commands
- A Gradio/demo behavior that exposes files, logs, secrets, or private metadata
- Dependency or packaging issues with realistic compromise risk
- Release assets that accidentally include private paths, secrets, or non-redistributable raw images
- Hugging Face token, API key, or deployment-secret exposure

## Model Safety Boundary

Reports about misclassification, bias, or weak model performance are important, but they are usually model-quality issues rather than security vulnerabilities. Use a public issue unless the report includes private data or would create realistic harm if disclosed immediately.

This project does not provide wildlife safety advice, species identification, or field-ready snake detection.

## How To Report

Use a private contact path from the maintainer's GitHub profile or portfolio site. Include:

- A short description of the issue
- Steps to reproduce
- Affected command, release asset, or demo URL
- Expected impact
- Whether private image data, exact locations, tokens, or local paths are involved

Please use synthetic images or public-domain examples where possible.

## Public Disclosure

Please give the maintainer time to confirm and fix security-sensitive issues before public disclosure. A public note may be added after a fix when it helps users understand the risk.
