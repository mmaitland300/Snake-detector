# Model Card: Snake Detector Real-Photo Demo

## Model Summary

| Field | Value |
| --- | --- |
| Model release | `v1.1.0-real-dataset` |
| Python package version | Separate package version in `pyproject.toml`; currently `0.1.0` |
| Task | Binary whole-image classification: `snake` vs `no_snake` |
| Model type | Keras image classifier using InceptionV3 preprocessing |
| Input size | 160 x 160 RGB image |
| Decision threshold | `0.76` for the `snake` class |
| Live demo | [mmaitland-snake-detector-demo.hf.space](https://mmaitland-snake-detector-demo.hf.space) |
| Release package | [v1.1.0-real-dataset](https://github.com/mmaitland300/Snake-detector/releases/tag/v1.1.0-real-dataset) |

The release model is used by the live Hugging Face Space. The lightweight
placeholder model and generated images in the repository are kept for
publication-safe local workflow checks and are not the live real-photo model.

## Intended Use

This model is intended for:

- A bounded portfolio demo of a real-photo computer-vision workflow.
- Demonstrating image preprocessing, model packaging, release assets, and a live
  Gradio deployment.
- Exploring a simple `snake` vs `no_snake` classifier on iNaturalist-sourced
  imagery.

It is not intended for:

- Species identification.
- Wildlife-handling, medical, rescue, or field-safety decisions.
- Confirming whether an image is safe to approach.
- Production monitoring or automated moderation.

## Training Data

The model was trained from an iNaturalist collection lane filtered to
CC0, CC-BY, and CC-BY-SA photo licenses. Raw third-party images are not
redistributed in this repository. See [DATASET_CARD.md](DATASET_CARD.md) and
[docs/dataset_and_license.md](docs/dataset_and_license.md).

Release manifest summary:

| Field | Value |
| --- | ---: |
| Manifest rows | 6,321 |
| `snake` rows | 1,271 |
| `no_snake` rows | 5,050 |
| Photo licenses included | CC0, CC-BY, CC-BY-SA |

Split summary:

| Split | `snake` | `no_snake` |
| --- | ---: | ---: |
| Training | 901 | 3,605 |
| Validation | 115 | 417 |
| Testing | 255 | 1,028 |

## Evaluation

Held-out test split: 1,283 iNaturalist-sourced images at threshold `0.76`.

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9026 |
| Macro precision | 0.8652 |
| Macro recall | 0.8139 |
| Macro F1 | 0.8358 |
| Snake precision | 0.8095 |
| Snake recall | 0.6667 |
| Snake F1 | 0.7312 |
| No-snake precision | 0.9208 |
| No-snake recall | 0.9611 |
| No-snake F1 | 0.9405 |

Confusion matrix at threshold `0.76`:

| Actual \ Predicted | `no_snake` | `snake` |
| --- | ---: | ---: |
| `no_snake` | 988 | 40 |
| `snake` | 85 | 170 |

Source: [docs/releases/v1.1.0-real-dataset.md](docs/releases/v1.1.0-real-dataset.md)

## Limitations

- The model is binary and does not identify species.
- The model can miss snakes; held-out snake recall is `0.6667`.
- Small, distant, partly hidden, blurred, or unusual images may be difficult.
- The negative class depends on the collected non-snake mix and should not be
  treated as exhaustive coverage of all real-world confusers.
- The model has not been validated for field safety, geographic robustness,
  camera-trap deployment, or medical/wildlife-handling use.
- The current public release documents one collected dataset lane and held-out
  split, not a broad external benchmark.

## Release Assets

The GitHub Release mirrors:

- `model.keras`
- run and deployment config
- held-out metrics
- validation threshold sweep
- confusion matrix
- training curves
- sample prediction summaries
- checksum manifest

The large model file belongs in GitHub Releases and Hugging Face Space storage,
not normal git history.

## Versioning Note

`v1.1.0-real-dataset` is the model/data release tag. It is intentionally
separate from the Python package version in `pyproject.toml`, which tracks the
installable CLI/package surface.
