# Dataset Provenance and Public Usage Rights

## Status

Portfolio-safe publication policy is active as of March 21, 2026.

Real-photo model release policy is active as of May 7, 2026: publish model/evaluation artifacts, not raw third-party image files.

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

### 3) iNaturalist real-photo collection

- Origin: iNaturalist observations collected through the repository CLI workflow
- Rights status: manifest filtered to CC0, CC-BY, and CC-BY-SA photo licenses
- Public policy: raw photos stay out of git and GitHub Releases; attribution/provenance stays in the local manifest
- Published assets: trained model, deployment config, run config, held-out metrics, threshold sweep, confusion matrix, and sample prediction summaries
- Intended purpose: live bounded demo and real-photo model evidence, not species identification or field-ready safety claims

## Repository Publication Rules

- Do not commit or publish raw scraped images unless redistribution rights are explicitly verified.
- Keep the original prototype corpus out of public-facing assets until each source is documented in `docs/dataset_sources.csv`.
- Use the generated placeholder dataset for lightweight public workflow proof.
- Use the iNaturalist lane only for documented model/evaluation artifact releases; do not publish raw third-party images without a separate redistribution review.
- Preserve the distinction between:
  - engineering validation on placeholder data
  - real-photo model evidence on a documented iNaturalist manifest

## Publication Checklist

- [x] Public-facing asset source is documented in `docs/dataset_sources.csv`.
- [x] The current public asset source allows redistribution.
- [x] `docs/attribution.md` reflects the current public asset policy.
- [x] README language avoids claiming public-image rights that are not verified.
- [x] Public benchmark and sample artifacts are generated only from approved placeholder data.
- [x] Real-photo model release documents the manifest summary and held-out metrics without redistributing raw images.
- [ ] Original prototype image sources individually verified for future public redistribution.

## Operational Guidance

- Keep raw real-image training corpora local and out of the git-tracked proof package unless redistribution is separately cleared.
- If you update the real-photo release package, update:
  - `docs/dataset_sources.csv`
  - `docs/attribution.md`
  - `README.md`
  - `docs/releases/<tag>.md`
  - any demo screenshots or sample prediction assets

## Current Honest Claim

This repository currently demonstrates a professionalized ML workflow and demo path in a legally safe way.
The live Space uses a real-photo iNaturalist-trained model, but it remains a bounded demo and does **not** claim production-ready wildlife safety or species identification.
