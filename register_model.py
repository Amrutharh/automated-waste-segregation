"""Phase 5 (registry) - register_model.py.

Registers the champion best_model.h5 in the MLflow Model Registry
(champion-challenger strategy). Usage: python register_model.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import BEST_MODEL_NAME, MODELS_DIR, REGISTERED_MODEL


def main(path=None, name=REGISTERED_MODEL):
    import mlflow
    path = path or os.path.join(MODELS_DIR, BEST_MODEL_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Champion not found: {path} (run train.py first)")
    mv = mlflow.register_model(model_uri=f"file://{os.path.abspath(path)}",
                               name=name)
    print(f"[registry] registered {name} version={mv.version} -> {path}")
    return mv


if __name__ == "__main__":
    main()
