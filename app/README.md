# Gradio Demo (Hugging Face Spaces)

## Local run

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -e .[demo]
.venv\Scripts\python app/gradio_app.py
```

The local app looks for `artifacts/model.joblib` first, then `artifacts/model.keras`.
Use the sklearn placeholder model for portfolio-safe validation, or provide a TensorFlow `.keras`
artifact if you later publish a rights-cleared model.
If you plan to run the TensorFlow `.keras` path, install ML extras as well: `pip install -e .[ml,demo]`.

For Keras models trained at a non-default size, keep `artifacts/run_config.json` next to the model (written by `train`) or set `SNAKE_DETECTOR_IMAGE_SIZE` when launching the demo.

## Hugging Face Spaces setup

1. Create a new Space with **Gradio** SDK.
2. Upload:
   - `app/gradio_app.py`
   - `requirements.txt` (or use this repo's optional dependency list)
   - model artifact (`artifacts/model.joblib` for the placeholder-safe baseline, or `artifacts/model.keras` for a TensorFlow build)
   - any approved demo images you want to showcase publicly.
3. Set Space hardware to CPU Basic (free tier).
4. Copy Space URL into your Vercel project page.

## Cold-start note

Free tier may sleep. Keep both:
- embedded iframe for in-page demo
- direct external "Open Demo" link as fallback
