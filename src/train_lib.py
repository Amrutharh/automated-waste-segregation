"""Phase 4 - Training & evaluation (PPT hyperparams).

Adam / categorical-crossentropy / accuracy, batch 32, up to 15 epochs,
early stopping + checkpoint (best_model.h5). Every run is MLflow-logged
(params, metrics, artifacts incl. accuracy/loss plots).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .augment import make_tf_dataset
from .config import (BATCH_SIZE, BEST_MODEL_NAME, CANDIDATES, CLASS_TO_ID,
                     EARLY_STOP_PATIENCE, EPOCHS, LEARNING_RATE,
                     MLFLOW_EXPERIMENT, MODELS_DIR)
from .data_loader import load_or_report
from .models import build_model, preprocess_for


def run_training(model_name="EfficientNetV2B0", epochs=EPOCHS,
                 batch=BATCH_SIZE, data_dir=None, subset=None,
                 experiment=MLFLOW_EXPERIMENT):
    import mlflow
    import tensorflow as tf
    from tensorflow import keras

    splits, info = load_or_report(data_dir)
    if subset:  # smoke-test shortcut: cap images per split
        splits = {k: v[:subset] for k, v in splits.items()}
    print({k: len(v) for k, v in splits.items()})

    prep = preprocess_for(model_name)
    train_ds = make_tf_dataset(splits["train"], CLASS_TO_ID, batch, True, prep)
    val_ds = make_tf_dataset(splits["val"], CLASS_TO_ID, batch, False, prep)
    test_ds = make_tf_dataset(splits["test"], CLASS_TO_ID, batch, False, prep)

    model = build_model(model_name)
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    ckpt = os.path.join(MODELS_DIR, f"ckpt_{model_name.lower()}.h5")
    cbs = [keras.callbacks.EarlyStopping(patience=EARLY_STOP_PATIENCE,
                                         restore_best_weights=True),
           keras.callbacks.ModelCheckpoint(ckpt, save_best_only=True)]

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({"model": model_name, "epochs": epochs,
                           "batch": batch, "lr": LEARNING_RATE,
                           "train_n": len(splits["train"]),
                           "real_data": info.get("real")})
        hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                         callbacks=cbs, verbose=1)
        loss, acc = model.evaluate(test_ds, verbose=0)
        mlflow.log_metrics({"test_loss": float(loss), "test_acc": float(acc),
                            "best_val_acc": float(max(hist.history["val_accuracy"]))})
        plot = save_curves(hist, model_name)
        mlflow.log_artifact(plot)
        mlflow.log_artifact(ckpt)
        mlflow.keras.log_model(model, artifact_path=model_name.lower())
    print(f"[train] {model_name}: test_acc={acc:.4f} -> {ckpt}")
    return {"model": model_name, "test_acc": float(acc),
            "test_loss": float(loss), "ckpt": ckpt}


def save_curves(hist, model_name):
    os.makedirs("reports", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist.history["accuracy"], label="train")
    ax[0].plot(hist.history["val_accuracy"], label="val")
    ax[0].set_title(f"{model_name} accuracy")
    ax[0].legend()
    ax[1].plot(hist.history["loss"], label="train")
    ax[1].plot(hist.history["val_loss"], label="val")
    ax[1].set_title(f"{model_name} loss")
    ax[1].legend()
    fig.tight_layout()
    p = os.path.join("reports", f"curves_{model_name.lower()}.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def compare_all(**kw):
    """Train every PPT candidate, copy the winner to best_model.h5."""
    import shutil
    results = [run_training(m, **kw) for m in CANDIDATES]
    best = max(results, key=lambda r: r["test_acc"])
    dst = os.path.join(MODELS_DIR, BEST_MODEL_NAME)
    shutil.copy(best["ckpt"], dst)
    print(f"[train] CHAMPION = {best['model']} ({best['test_acc']:.4f}) -> {dst}")
    return best, results
