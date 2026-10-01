"""Shared inference helpers (used by FastAPI + Streamlit demo).

TF-dependent pieces are injectable so this module imports everywhere.
"""
import numpy as np
from PIL import Image

from .config import CLASSES, IMG_SIZE, LOW_CONF_THRESHOLD


def preprocess_pil(img: Image.Image, preprocess_fn, size=IMG_SIZE):
    arr = np.asarray(img.convert("RGB").resize(size), dtype=np.float32)
    return preprocess_fn(np.expand_dims(arr, 0))


def predict_array(model, batch):
    proba = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(proba))
    conf = float(proba[idx])
    return CLASSES[idx], conf, {c: round(float(p), 4) for c, p in zip(CLASSES, proba)}


def predict_image(model, img: Image.Image, preprocess_fn):
    label, conf, scores = predict_array(model, preprocess_pil(img, preprocess_fn))
    return {"predicted_class": label, "confidence": round(conf, 4),
            "manual_review": conf < LOW_CONF_THRESHOLD, "all_scores": scores}
