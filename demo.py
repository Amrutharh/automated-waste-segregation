"""Demo website - upload a waste photo, get class + confidence.
Run:  streamlit run demo.py   (needs models/best_model.h5 from training)
"""
import os
import sys

import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import BEST_MODEL_NAME, CLASSES, MODELS_DIR
from src.inference import predict_image

st.set_page_config(page_title="Waste Segregation Demo", page_icon="♻️",
                   layout="centered")

st.markdown(
    """<div style="background:linear-gradient(120deg,#059669,#0EA5E9);
    border-radius:16px;padding:24px 28px;color:white;margin-bottom:16px">
    <h1 style="margin:0">♻️ Waste Segregation Demo</h1>
    <p style="margin:4px 0 0 0">EfficientNetV2B0 · 6 classes · confidence +
    manual-review flag</p></div>""",
    unsafe_allow_html=True,
)

BIN_COLOR = {"cardboard": "🟫", "glass": "⬜", "metal": "⬛",
             "paper": "⬜", "plastic": "🟦", "trash": "🟥"}


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    from tensorflow import keras
    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
    path = os.path.join(MODELS_DIR, BEST_MODEL_NAME)
    if not os.path.exists(path):
        return None, None
    return keras.models.load_model(path), preprocess_input


model, preprocess = get_model()
if model is None:
    st.warning("No trained model yet (`models/best_model.h5` missing). "
               "Train on Colab GPU: `python train.py --model all --epochs 15`, "
               "then place the file in `models/`. Showing UI preview below.")
    st.info("Classes: " + ", ".join(CLASSES))
    st.stop()

up = st.file_uploader("Upload a waste photo", type=["jpg", "jpeg", "png", "webp"])
if up is None:
    st.caption("Waiting for an image...")
    st.stop()

img = Image.open(up)
st.image(img, caption="Input", use_container_width=True)
with st.spinner("Classifying..."):
    out = predict_image(model, img, preprocess)

st.metric("Predicted class",
          f"{BIN_COLOR.get(out['predicted_class'], '')} {out['predicted_class']}")
st.metric("Confidence", f"{out['confidence']:.0%}")
st.progress(out["confidence"], text="Model confidence")
if out["manual_review"]:
    st.warning("⚠️ Low confidence - flag for manual review.")
else:
    st.success("High-confidence prediction.")
with st.expander("All class scores"):
    st.bar_chart(out["all_scores"])
