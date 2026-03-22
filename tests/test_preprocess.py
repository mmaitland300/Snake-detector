import numpy as np

from snake_detector.data import preprocess_pixels_inception


def test_preprocess_pixels_range() -> None:
    img = np.array([[[0, 127.5, 255]]], dtype=np.float32)
    out = preprocess_pixels_inception(img)
    assert np.isclose(out.min(), -1.0)
    assert np.isclose(out.max(), 1.0)
