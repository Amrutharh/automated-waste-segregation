"""Phase 3 - Model development (TensorFlow/Keras, PPT specs).

Backbones (ImageNet, frozen): EfficientNetV2B0 (champion), ResNet50,
MobileNetV2. Custom head: GAP -> Dense(256, ReLU) -> Dropout(0.3) ->
Softmax(6). TF imports are lazy so data/API/CI code runs without TF.
"""
from .config import CLASSES, DENSE_UNITS, DROPOUT


def preprocess_for(name):
    from tensorflow import keras
    if name == "EfficientNetV2B0":
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
    elif name == "ResNet50":
        from tensorflow.keras.applications.resnet import preprocess_input
    else:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    return preprocess_input


def build_model(name="EfficientNetV2B0", n_classes=len(CLASSES),
                dense_units=DENSE_UNITS, dropout=DROPOUT):
    from tensorflow import keras
    from tensorflow.keras import layers

    if name == "EfficientNetV2B0":
        from tensorflow.keras.applications import EfficientNetV2B0 as Net
    elif name == "ResNet50":
        from tensorflow.keras.applications import ResNet50 as Net
    elif name == "MobileNetV2":
        from tensorflow.keras.applications import MobileNetV2 as Net
    else:
        raise ValueError(f"Unknown backbone {name}")

    base = Net(weights="imagenet", include_top=False,
               input_shape=(224, 224, 3))
    base.trainable = False  # frozen per PPT
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(base.input, out, name=name.lower())
    model.backbone_name = name
    return model
