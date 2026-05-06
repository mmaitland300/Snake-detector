# Deployment Decision - Snake Detector Demo

## Status

Finalized on March 21, 2026

## Context

- Portfolio site is hosted on Vercel.
- The repository now has a validated local Gradio demo path and a reproducible placeholder-safe model artifact.
- A public Hugging Face Space is now published and linked from the portfolio. The hosted demo is still a bounded snake/no-snake experiment backed by the current public artifact path, not a field-safe wildlife classifier.
- The model artifact needs a single documented public hosting choice for release packaging.

## Final Decision

Use a two-surface delivery model:

1. Demo surface: Hugging Face Spaces (Gradio) at the published public Space URL (same bounded demo as local).
2. Portfolio and artifact surface: GitHub repository + **GitHub Release asset** for the pinned model artifact.

## Why GitHub Release Asset Won

- It keeps the artifact attached to the same tagged portfolio release as the docs and proof package.
- The validated placeholder-safe baseline artifact is small enough to distribute this way.
- It avoids introducing a second remote model registry: the hosted demo consumes the documented GitHub Release artifact path instead of HF Model Hub weights.
- It matches the current evidence level: local and hosted demos exercise the same pinned release artifact, while public benchmarking stays on generated placeholder data.

## Current Publishable State

- Local demo path: validated
- Public benchmark artifacts: generated
- Public Space URL: https://huggingface.co/spaces/mmaitland/snake-detector-demo
- Safe public artifact to ship first: `artifacts/model.joblib`

## Fallback Plan if the Hugging Face Space is unavailable

If the Space is unavailable, the portfolio page should show:

- benchmark table
- confusion matrix
- sample prediction panel
- short note that the public benchmark is on generated placeholder data
- local-demo validation instructions or a later demo screenshot/GIF once captured

## Non-goals for This Release

- Claiming field-ready wildlife monitoring, species identification, or real-world performance beyond the published bounded demo
- Shipping unverified scraped images
- Presenting the placeholder benchmark as a real-world wildlife-performance claim

## Revisit Triggers

- The Hugging Face Space URL, app endpoint (`*.hf.space`), or portfolio `NEXT_PUBLIC_SNAKE_DEMO_URL` changes and docs must stay in sync
- A rights-cleared real-image dataset becomes available for public benchmarking
- The artifact outgrows practical GitHub Release distribution
