# Demo Capture Checklist

- [ ] Capture one screenshot of the upload form.
- [ ] Capture one screenshot of a successful `snake` prediction.
- [ ] Capture one screenshot of a `no_snake` prediction.
- [ ] Optionally record a 5-10 second GIF walkthrough.

Current state:
- Local Gradio app is wired to `artifacts/model.joblib` and ready for manual browser capture.
- Local validation evidence on March 21, 2026:
  - `build_demo()` returned a Gradio `Blocks` app
  - `http://127.0.0.1:7860/` responded with HTTP 200 during startup validation
  - `_predict()` returned `snake` with `0.9902` confidence for `data/public_demo_split/testing/snake/snake_039.png`
- A public Hugging Face Space URL is not claimed yet.
- Automated screenshot capture is still pending because local headless Chrome failed with a Crashpad access-denied error in this workspace.

Store final media as:
- `docs/assets/demo_screenshot.png` and/or
- `docs/assets/demo_walkthrough.gif`
