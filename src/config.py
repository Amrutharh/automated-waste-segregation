"""Phase 0 - Config. Single source of truth (mirrors the PPT methodology)."""
import os

# 6 waste classes (PPT Phase 1)
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# Preprocessing (PPT Phase 2)
IMG_SIZE = (224, 224)
NORMALIZE = "efficientnet"  # efficientnet.preprocess_input (matches training)

# Splits (PPT: train 70 / val 20 / test 10)
SPLITS = {"train": 0.7, "val": 0.2, "test": 0.1}
SEED = 42

# Training hyperparams (PPT Phase 4 / Results)
OPTIMIZER = "adam"
LEARNING_RATE = 1e-3
LOSS = "categorical_crossentropy"
BATCH_SIZE = 32
EPOCHS = 15
EARLY_STOP_PATIENCE = 3
DROPOUT = 0.3
DENSE_UNITS = 256

# Model comparison set (PPT Phase 3) - first is the champion
CANDIDATES = ["EfficientNetV2B0", "ResNet50", "MobileNetV2"]

# Paths
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")
REPORTS_DIR = os.path.join(BASE, "reports")
LOGS_DIR = os.path.join(BASE, "logs")
BEST_MODEL_NAME = "best_model.h5"

# Where the real images live (class-wise folders). Accepts TrashNet layout,
# Garbage-Classification layout, or your manually collected folders.
DATA_CANDIDATES = [
    os.path.join(DATA_DIR, "TrashNet"),
    os.path.join(DATA_DIR, "Manual"),  # manually/publicly collected extra images
    os.path.join(DATA_DIR, "dataset"),
    os.path.join(DATA_DIR, "Garbage classification"),
    DATA_DIR,
]

# MLOps
MLFLOW_EXPERIMENT = "waste-segregation"
REGISTERED_MODEL = "waste-classifier"
LOW_CONF_THRESHOLD = 0.6  # below this -> flag for manual review
