# Deployment Decision - Snake Detector Demo

## Status

Finalized on March 21, 2026

## Context

- Portfolio site is hosted on Vercel.
- The repository now has a validated local Gradio demo path and a reproducible placeholder-safe model artifact.
- A public Hugging Face Space has not been published from this workspace, so the repo must not claim a live URL yet.
- The model artifact needs a single documented public hosting choice for release packaging.

## Final Decision

Use a two-surface delivery model:

1. Demo surface: Hugging Face Spaces (Gradio) when manually published.
2. Portfolio and artifact surface: GitHub repository + **GitHub Release asset** for the pinned model artifact.

## Why GitHub Release Asset Won

- It keeps the artifact attached to the same tagged portfolio release as the docs and proof package.
- The validated placeholder-safe baseline artifact is small enough to distribute this way.
- It avoids introducing a second remote model registry before a real public Space is published.
- It matches the current evidence level: local validation is complete, remote deployment is still a manual follow-up.

## Current Publishable State

- Local demo path: validated
- Public benchmark artifacts: generated
- Public Space URL: not yet published
- Safe public artifact to ship first: `artifacts/model.joblib`

## Fallback Plan

Until a Hugging Face Space is live, the portfolio page should show:

- benchmark table
- confusion matrix
- sample prediction panel
- short note that the public benchmark is on generated placeholder data
- local-demo validation instructions or a later demo screenshot/GIF once captured

## Non-goals for This Release

- Claiming an active hosted URL that has not been published
- Shipping unverified scraped images
- Presenting the placeholder benchmark as a real-world wildlife-performance claim

## Revisit Triggers

- A Hugging Face Space is published and needs a permanent URL added to the README
- A rights-cleared real-image dataset becomes available for public benchmarking
- The artifact outgrows practical GitHub Release distribution
