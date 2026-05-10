# Roadmap

This roadmap keeps Snake Detector's direction realistic. The project is a bounded `snake` vs `no_snake` image-classification demo with a live Hugging Face app, package-based CLI, and documented model/dataset limits.

## Current Focus

- Keep the live Hugging Face demo aligned with the documented model release.
- Keep model and dataset claims bounded by the model card, dataset card, and release package.
- Preserve reproducible local workflows without committing raw third-party images.
- Keep placeholder-safe engineering artifacts clearly separate from real-photo model assets.

## Near-Term Work

- Add more tests around CLI argument validation and artifact loading.
- Improve docs for reproducing real-photo training from a rights-aware manifest.
- Keep release notes synchronized with model artifacts, metrics, thresholds, and demo config.
- Add clearer troubleshooting notes for local Gradio and Hugging Face Spaces behavior.
- Continue tightening dataset/source documentation and attribution posture.

## Model And Evaluation Work

- Add comparison notes when future model releases change input size, threshold, or preprocessing.
- Track false-positive and false-negative examples without publishing private or restricted images.
- Keep species-identification and field-safety claims out of scope unless the project is redesigned around those goals.
- Document any future human-review or external validation separately from local engineering benchmarks.

## Not Planned Right Now

- Species-level classification
- Field-ready wildlife safety guarantees
- Mobile app deployment
- Raw third-party image redistribution in git
- Broad ecological or conservation claims beyond the evaluated binary demo

## Contribution Fit

Good contributions are small, reproducible, and aligned with the current package/demo/release workflow. See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.
