# Dataset Card: Snake Detector Data Lanes

## Dataset Summary

Snake Detector tracks three data lanes separately so public claims stay clear:

| Lane | Public status | Purpose |
| --- | --- | --- |
| Original prototype corpus | Local-only; redistribution rights not verified | Historical experimentation only |
| Generated placeholder dataset | Safe to regenerate and redistribute | Lightweight local workflow and checked-in README assets |
| iNaturalist real-photo collection | Raw photos stay out of git; model/evaluation artifacts are released | Live real-photo demo model and held-out metrics |

The real-photo release described by `v1.1.0-real-dataset` is based on the
iNaturalist lane. The raw third-party images are not committed to git.

## Real-Photo Release Dataset

| Field | Value |
| --- | --- |
| Release | `v1.1.0-real-dataset` |
| Provider | iNaturalist |
| Task | Binary `snake` vs `no_snake` image classification |
| License filter | CC0, CC-BY, CC-BY-SA |
| Raw photo redistribution | Not included in git or release package |
| Published artifacts | Model, configs, metrics, threshold sweep, confusion matrix, sample summaries |

Manifest summary:

| Field | Value |
| --- | ---: |
| Manifest rows | 6,321 |
| `snake` rows | 1,271 |
| `no_snake` rows | 5,050 |

Split summary:

| Split | `snake` | `no_snake` |
| --- | ---: | ---: |
| Training | 901 | 3,605 |
| Validation | 115 | 417 |
| Testing | 255 | 1,028 |

Primary documentation:

- [docs/releases/v1.1.0-real-dataset.md](docs/releases/v1.1.0-real-dataset.md)
- [docs/real_image_collection.md](docs/real_image_collection.md)
- [docs/collection_runs.md](docs/collection_runs.md)
- [docs/dataset_and_license.md](docs/dataset_and_license.md)
- [docs/attribution.md](docs/attribution.md)

## Collection Method

The iNaturalist lane uses the repository CLI to:

- Resolve taxa through iNaturalist taxon lookup.
- Collect observation/photo metadata into a manifest.
- Filter to approved photo licenses.
- Download images locally for training and evaluation.
- Keep raw image files and bulk split directories out of git.

The no-snake side is intentionally collected from multiple non-snake taxa rather
than only blank backgrounds, so the model sees more realistic visual confusers.

## Placeholder Dataset

The placeholder dataset is generated locally with:

```bash
python -m snake_detector.demo_data
```

It is used for:

- Lightweight local CLI checks.
- Publication-safe README assets.
- A reproducible baseline that does not redistribute third-party wildlife
  photos.

It is not the live real-photo model dataset.

## Licensing And Attribution

- Repository code is MIT licensed.
- Generated placeholder images are repository-created and safe for public
  examples.
- iNaturalist photo metadata is filtered to CC0, CC-BY, and CC-BY-SA.
- Raw iNaturalist photos are not redistributed in this repository.
- Per-image attribution/provenance remains in the local manifest rather than
  being copied into git as bulk image data.

If raw third-party imagery is ever published later, required attribution and
redistribution rights must be reviewed and documented first.

## Known Limitations

- The real-photo dataset is suitable for a bounded demo, not field-safety
  validation.
- The collection is not a comprehensive global snake/no-snake benchmark.
- The no-snake class is a sampled negative mix, not every possible non-snake
  object or habitat.
- iNaturalist imagery may reflect observer, geography, species, camera, and
  composition biases.
- The current release does not document multi-site, multi-camera, or deployment
  environment robustness.

## Versioning Note

`v1.1.0-real-dataset` names the real-photo model/dataset release. It is separate
from the Python package version in `pyproject.toml`.
