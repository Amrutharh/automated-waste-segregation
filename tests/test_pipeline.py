"""Smoke tests (run without GPU/TF): python -m pytest tests/ -v"""
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image


def test_split_ratios_and_clean():
    from src.data_loader import clean_items, distribution, stratified_split, synthetic_items
    items = synthetic_items(n_per_class=20)
    # inject a duplicate + a corrupt file
    items = items + [items[0]]
    bad = os.path.join(os.path.dirname(items[0][0]), "bad.jpg")
    open(bad, "w").write("not an image")
    items.append((bad, "plastic"))
    good, dropped = clean_items(items)
    assert dropped["duplicate"] >= 1 and dropped["corrupt"] >= 1
    sp = stratified_split(good)
    n = len(good)
    assert abs(len(sp["train"]) / n - 0.7) < 0.05
    assert abs(len(sp["val"]) / n - 0.2) < 0.05
    assert set(distribution(good)) == {"cardboard", "glass", "metal",
                                       "paper", "plastic", "trash"}


def test_api_predict_with_stub_model():
    import app_mp
    from fastapi.testclient import TestClient

    class Stub:
        def predict(self, arr, verbose=0):
            p = np.zeros((1, 6))
            p[0, 4] = 0.93  # plastic
            return p

    app_mp.get_model = lambda: Stub()
    app_mp._PREPROCESS = lambda x: x / 255.0
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), (200, 50, 50)).save(buf, format="JPEG")
    r = TestClient(app_mp.app).post(
        "/predict", files={"file": ("bottle.jpg", buf.getvalue(), "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predicted_class"] == "plastic"
    assert body["confidence"] > 0.9 and body["manual_review"] is False
