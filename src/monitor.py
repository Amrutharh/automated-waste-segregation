"""Phase 9 - Monitoring & logging helpers.

Training metrics, prediction outputs (+confidence), and error/system
logs. JSONL files under logs/ for performance tracking.
"""
import json
import os
import traceback
from datetime import datetime

from .config import LOGS_DIR

os.makedirs(LOGS_DIR, exist_ok=True)
_PRED_LOG = os.path.join(LOGS_DIR, "predictions.jsonl")
_ERR_LOG = os.path.join(LOGS_DIR, "errors.log")
_TRAIN_LOG = os.path.join(LOGS_DIR, "training.jsonl")


def _append(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def log_training(result: dict):
    _append(_TRAIN_LOG, {"ts": datetime.now().isoformat(timespec="seconds"),
                         **result})


def log_prediction(image_name: str, pred: str, conf: float, flagged: bool):
    _append(_PRED_LOG, {"ts": datetime.now().isoformat(timespec="seconds"),
                        "image": image_name, "pred": pred,
                        "conf": round(float(conf), 4),
                        "manual_review": flagged})


def log_error(context: str, exc: Exception):
    with open(_ERR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {context}: "
                f"{exc!r}\n{traceback.format_exc()}\n")
