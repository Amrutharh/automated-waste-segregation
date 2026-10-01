# ♻️ Automated Waste Segregation using Deep Learning + MLOps

**🚀 Live Demo:** https://automated-waste-segregation-npiekcvayhdvzzbavnqnir.streamlit.app/

6-class waste classifier (**EfficientNetV2B0**, transfer learning) with a full
MLOps pipeline: MLflow tracking + registry, GitHub Actions CI, FastAPI
inference API, monitoring logs. Built to match the project PPT phase-by-phase.

## PPT phase map

| PPT phase | What | File |
|---|---|---|
| 1 Collection | TrashNet + Garbage + manual folders; dedup + corrupt removal | `src/data_loader.py` |
| 2 Preprocess/EDA | 224×224 RGB, EfficientNet norm, 70/20/10 stratified split | `src/data_loader.py`, `src/augment.py` |
| 3 Model dev | EfficientNetV2B0 (frozen) + GAP/Dense-256/ReLU/Dropout-0.3/Softmax-6; ResNet50 + MobileNetV2 compared | `src/models.py` |
| 4 Training | Adam, categorical-CE, bs32, ≤15 epochs, early stopping, checkpoint | `train.py`, `src/train_lib.py` |
| 5 MLOps | MLflow params/metrics/artifacts, model registry (champion-challenger) | `register_model.py` |
| 6 CI/CD | Push-triggered env setup, validation, tests | `.github/workflows/mlops_mp.yml` |
| 7 Deploy | FastAPI `POST /predict` → class + confidence | `app_mp.py` |
| 8 Debug | Same-class bug guard: matched train/inference preprocessing | `src/augment.py`, `src/models.py` |
| 9 Monitor | Training/prediction/error logs (`logs/`) | `src/monitor.py` |

```
images → clean/dedup → 70/20/10 split → Albumentations → EfficientNetV2B0 →
MLflow log → register champion → FastAPI /predict → logs
```

## Quickstart

```bash
pip install -r requirements.txt        # + tensorflow + albumentations for training

# 1. (optional) drop real images under data/  (see data/README.md)
# 2. smoke test the pipeline (no GPU/TF needed)
python -m pytest tests/ -v

# 3. train champion (needs TF; GPU recommended, 16k imgs)
python train.py --model all --epochs 15

# 4. register + serve
python register_model.py
uvicorn app_mp:app --reload
# POST an image to http://localhost:8000/predict
```

Reported reference: **91.92% val accuracy @ epoch 12** (EfficientNetV2B0).
