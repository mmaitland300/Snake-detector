# Deployment Decision - Snake Detector Demo

## Status

Finalized on March 21, 2026. Updated on May 7, 2026 after the live Hugging Face Space moved to a real-photo iNaturalist-trained Keras model.

## Context

- Portfolio site is hosted on Vercel.
- Public demo is hosted on Hugging Face Spaces.
- The live Space now serves a real-photo Keras model (`model.keras`) with `image_size=160`, `threshold=0.76`, and InceptionV3 preprocessing.
- The repository still keeps a lightweight placeholder-safe baseline (`artifacts/model.joblib`) for reproducible engineering proof and local smoke tests.
- The real Keras model is too large for normal git history, so it needs release/Space storage instead of direct repository tracking.

## Current Decision

Use a three-surface delivery model:

1. **Demo surface:** Hugging Face Spaces (Gradio), with portfolio links using `https://mmaitland-snake-detector-demo.hf.space`.
2. **Large-artifact mirror:** GitHub Release `v1.1.0-real-dataset`, containing `model.keras`, deployment config, run config, held-out metrics, threshold sweep, and visual evaluation artifacts.
3. **Repository proof surface:** README/docs/tests/CI plus the smaller placeholder-safe baseline that demonstrates the package workflow without redistributing third-party photos.

## Why This Shape

- Hugging Face Space storage is the practical runtime home for the live Keras model.
- GitHub Releases are appropriate for versioned model artifacts that are too large for normal git tracking.
- Keeping placeholder artifacts labeled as an engineering baseline avoids mixing legal-safe workflow proof with real-photo model-quality claims.
- The real-photo release package lets reviewers verify what is live without requiring raw iNaturalist image redistribution.

## Current Publishable State

- Live app URL: `https://mmaitland-snake-detector-demo.hf.space`
- Space repo URL: `https://huggingface.co/spaces/mmaitland/snake-detector-demo`
- Live model: Space-side `model.keras`
- Release package: [releases/v1.1.0-real-dataset.md](releases/v1.1.0-real-dataset.md)
- Older placeholder artifact: [GitHub Release v1.0.0 `model.joblib`](https://github.com/mmaitland300/Snake-detector/releases/download/v1.0.0/model.joblib)

## Fallback Plan if the Hugging Face Space Is Unavailable

If the Space is unavailable, the portfolio page should show:

- Real-photo release metrics and model artifact link
- Placeholder-safe workflow benchmark, clearly labeled as a baseline
- Confusion matrix and sample prediction artifacts
- Local-demo validation instructions
- A short note that the app is a bounded binary demo, not wildlife safety guidance

## Non-Goals for This Release

- Claiming field-ready wildlife monitoring, species identification, or safe handling guidance
- Shipping raw third-party images in git or GitHub Releases
- Presenting placeholder benchmark metrics as real-world wildlife performance
- Comparing placeholder metrics directly against real-photo metrics as if they came from the same dataset

## Revisit Triggers

- The Hugging Face Space URL, app endpoint, or portfolio `NEXT_PUBLIC_SNAKE_DEMO_URL` changes
- The Space runtime is mirrored back to GitHub and should download the Keras release asset automatically
- A larger or cleaner rights-audited real-image dataset is prepared
- The model artifact outgrows practical GitHub Release distribution
