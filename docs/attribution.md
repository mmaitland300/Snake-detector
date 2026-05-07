# Attribution

As of May 7, 2026, this repository does **not** redistribute raw third-party wildlife photos.

Public-facing image assets in the repo currently come from the generated placeholder dataset created by
`python -m snake_detector.demo_data`, so no external image attribution is required for the checked-in
benchmark, confusion matrix, sample predictions, or demo-safe examples.

The live Hugging Face Space uses a model trained from an iNaturalist manifest filtered to CC0, CC-BY,
and CC-BY-SA photo licenses. The release publishes the trained model and evaluation artifacts, not the
raw photos. Per-image attribution/provenance is retained in the local collection manifest rather than
copied into git as bulk image data.

If raw third-party imagery is added later, append the required attribution text here and mirror the same
text in `README.md` before publishing those assets.
