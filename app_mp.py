"""Phase 7 - FastAPI deployment (app_mp.py).

Endpoints: GET /health, POST /predict (image -> class + confidence).
Low-confidence predictions are flagged for manual review (PPT).
Run:  uvicorn app_mp:app --reload   (from project root)
"""
import io
import os
import sys

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (BEST_MODEL_NAME, CLASSES, IMG_SIZE,
                        LOW_CONF_THRESHOLD, MODELS_DIR)
from src.monitor import log_error, log_prediction

app = FastAPI(title="Waste Segregation API",
              description="EfficientNetV2B0 classifier: image -> class + confidence")
_MODEL = None
_PREPROCESS = None


def get_model():
    """Lazy load best_model.h5 (EfficientNet preprocessing)."""
    global _MODEL, _PREPROCESS
    if _MODEL is None:
        from tensorflow import keras
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
        path = os.path.join(MODELS_DIR, BEST_MODEL_NAME)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} missing - run train.py (or tests ship a dummy model).")
        _MODEL = keras.models.load_model(path)
        _PREPROCESS = preprocess_input
    return _MODEL


@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASSES,
            "model_loaded": _MODEL is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize(IMG_SIZE)
        arr = _PREPROCESS(np.expand_dims(np.asarray(img, dtype=np.float32), 0))
        proba = get_model().predict(arr, verbose=0)[0]
        idx = int(np.argmax(proba))
        conf = float(proba[idx])
        flagged = conf < LOW_CONF_THRESHOLD
        log_prediction(file.filename, CLASSES[idx], conf, flagged)
        return {"predicted_class": CLASSES[idx], "confidence": round(conf, 4),
                "manual_review": flagged,
                "all_scores": {c: round(float(p), 4) for c, p in zip(CLASSES, proba)}}
    except Exception as exc:  # Phase 8: never crash the API, log instead
        log_error("predict", exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})
