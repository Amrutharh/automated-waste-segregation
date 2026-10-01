"""Augmentation (PPT: Albumentations) + tf.data pipelines (224x224 RGB).

Geometric: flips + rotate-90. Photometric: brightness/contrast.
Noise: Gaussian (edge-camera simulation). Falls back to plain
Keras preprocessing if albumentations is not installed.
"""
import numpy as np
from PIL import Image

from .config import BATCH_SIZE, IMG_SIZE

try:
    import albumentations as A

    def train_augmenter():
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
        ])
    HAS_ALBU = True
except Exception:
    HAS_ALBU = False

    def train_augmenter():
        return None  # identity fallback


def load_rgb(path, size=IMG_SIZE):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize(size))


def augment_array(arr):
    aug = train_augmenter()
    if aug is None:
        return arr
    return aug(image=arr)["image"]


def make_tf_dataset(items, class_to_id, batch=BATCH_SIZE, training=False,
                    preprocess_fn=None):
    """items: [(path, class_name)]. Returns (tf.data.Dataset, steps)."""
    import tensorflow as tf

    paths = [p for p, _ in items]
    labels = [class_to_id[c] for _, c in items]

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.uint8)
        if training:
            img = tf.numpy_function(augment_array, [img], tf.uint8)
            img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
        if preprocess_fn is not None:
            img = preprocess_fn(tf.cast(img, tf.float32))
        else:
            img = tf.cast(img, tf.float32) / 255.0
        return img, tf.one_hot(label, len(class_to_id))

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(items), 2048))
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds
