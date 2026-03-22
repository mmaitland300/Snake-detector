# Dataset Provenance and Public Usage Rights

## Status

Portfolio-safe publication policy is active as of March 21, 2026.

## Dataset Lanes

### 1) Original prototype corpus

- Origin: historical web-scraped prototype data
- Rights status: not verified for redistribution
- Public policy: excluded from the repository, README examples, and demo screenshots
- Allowed use today: private/local experimentation only, pending per-source legal review

### 2) Approved public-safe placeholder dataset

- Origin: generated locally with `python -m snake_detector.demo_data`
- Rights status: safe to redistribute as repository-generated placeholder content
- Public policy: allowed for README visuals, benchmark artifacts, local demo validation, and publication-safe examples
- Intended purpose: validate the engineering workflow without publishing third-party wildlife imagery

## Repository Publication Rules

- Do not commit or publish raw scraped images unless redistribution rights are explicitly verified.
- Keep the original prototype corpus out of public-facing assets until each source is documented in `docs/dataset_sources.csv`.
- Use the generated placeholder dataset for public proof artifacts unless and until a rights-cleared real-image dataset is documented.
- Preserve the distinction between:
  - engineering validation on placeholder data
  - future model-quality claims on licensed real-world imagery

## Publication Checklist

- [x] Public-facing asset source is documented in `docs/dataset_sources.csv`.
- [x] The current public asset source allows redistribution.
- [x] `docs/attribution.md` reflects the current public asset policy.
- [x] README language avoids claiming public-image rights that are not verified.
- [x] Public benchmark and sample artifacts are generated only from approved placeholder data.
- [ ] Original prototype image sources individually verified for future public redistribution.

## Operational Guidance

- Keep any unverified real-image training corpus local and out of the git-tracked proof package.
- If you later replace the placeholder benchmark with a licensed real-image benchmark, update:
  - `docs/dataset_sources.csv`
  - `docs/attribution.md`
  - `README.md`
  - any demo screenshots or sample prediction assets

## Current Honest Claim

This repository currently demonstrates a professionalized ML workflow and demo path in a legally safe way.
It does **not** claim that the checked-in benchmark represents production-ready performance on licensed snake photography.
