"""Waste Segregation Demo - paged wow app.
Page 1 About -> Page 2 Upload & classify -> Page 3 Confidence deep-dive.
Run:  streamlit run demo.py   (needs models/best_model.h5 from training)
"""
import os
import sys

import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import BEST_MODEL_NAME, CLASSES, MODELS_DIR
from src.inference import predict_image

st.set_page_config(page_title="Waste Segregation AI", page_icon="♻️",
                   layout="wide")

st.markdown(
    """
<style>
.hero {
  background: linear-gradient(120deg, #059669 0%, #0EA5E9 60%, #6366F1 100%);
  border-radius: 16px; padding: 24px 28px; margin-bottom: 16px; color: white;
}
.hero h1 { margin: 0 0 6px 0; font-size: 2rem; color: white; }
.hero p { margin: 2px 0; opacity: 0.93; color: white; }
.bincard {
  background: #151F35; border: 1px solid #263252;
  border-radius: 12px; padding: 14px 16px; text-align: center;
  margin-bottom: 10px;
}
.bincard .big { font-size: 2rem; }
.bincard .name { font-weight: 700; margin-top: 4px; }
.step {
  background: #151F35; border: 1px solid #263252;
  border-radius: 12px; padding: 16px 18px; margin-bottom: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)

BIN_META = {
    "cardboard": ("📦", "Cardboard", "Boxes, cartons - recycle flat."),
    "glass": ("🍾", "Glass", "Bottles, jars - rinse first."),
    "metal": ("🥫", "Metal", "Cans, foils - recycle."),
    "paper": ("📰", "Paper", "Newspaper, sheets - keep dry."),
    "plastic": ("🧴", "Plastic", "Bottles, containers - check code."),
    "trash": ("🗑️", "Trash", "Non-recyclable - landfill."),
}

PAGES = ["1 · About", "2 · Classify", "3 · Confidence"]
if "page" not in st.session_state:
    st.session_state.page = 0


def goto(i):
    st.session_state.page = max(0, min(len(PAGES) - 1, i))


@st.cache_resource(show_spinner="Loading champion model...")
def get_model():
    from tensorflow import keras
    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
    path = os.path.join(MODELS_DIR, BEST_MODEL_NAME)
    if not os.path.exists(path):
        return None, None
    return keras.models.load_model(path), preprocess_input


model, preprocess = get_model()

st.markdown(
    """<div class="hero">
<h1>♻️ Waste Segregation AI</h1>
<p>EfficientNetV2B0 champion (test accuracy 92.3%) · 6 bins ·
confidence + manual-review flag · MLflow + FastAPI backend.</p>
</div>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Navigate")
    choice = st.radio("Go to", PAGES, index=st.session_state.page,
                      label_visibility="collapsed")
    goto(PAGES.index(choice))
    st.divider()
    st.header("Champion")
    st.success("EfficientNetV2B0 · test acc **92.3%**")
    st.caption("Runner-up MobileNetV2 · 88.7%. Full comparison in training logs.")
    st.caption("Tip: low-confidence items are flagged for manual review.")

page = st.session_state.page

# ---------------------------------------------------------- Page 1 About ---
if page == 0:
    st.subheader("What is this?")
    st.write("Upload a photo of waste. The AI puts it in the right bin - "
             "cardboard, glass, metal, paper, plastic or trash - and tells you "
             "how sure it is. Unsure items get flagged for a human to check.")
    st.markdown("### The 6 bins")
    cols = st.columns(3)
    for i, cls in enumerate(CLASSES):
        emoji, name, tip = BIN_META[cls]
        with cols[i % 3]:
            st.markdown(
                f"<div class='bincard'><div class='big'>{emoji}</div>"
                f"<div class='name'>{name}</div><div>{tip}</div></div>",
                unsafe_allow_html=True,
            )
    st.markdown("### How it works (3 steps)")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("<div class='step'>**1 · Upload**<br/>Snap or pick a waste photo.</div>",
                    unsafe_allow_html=True)
    with s2:
        st.markdown("<div class='step'>**2 · AI classifies**<br/>224×224 EfficientNet inference in milliseconds.</div>",
                    unsafe_allow_html=True)
    with s3:
        st.markdown("<div class='step'>**3 · Trust the score**<br/>High confidence bins it. Low confidence flags a human.</div>",
                    unsafe_allow_html=True)
    st.button("Start: classify a photo →", on_click=goto, args=(1,),
              type="primary")

# ------------------------------------------------------ Page 2 Classify ---
elif page == 1:
    st.subheader("Upload a waste photo")
    if model is None:
        st.warning("Champion model not deployed yet (`models/best_model.h5` missing). "
                   "The pipeline, API and tests all run without it.")
        st.stop()
    up = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
    if up is None:
        st.caption("Waiting for an image... (try one of your metal photos)")
        st.stop()
    img = Image.open(up)
    st.image(img, caption="Input image", use_container_width=True)
    with st.spinner("Classifying..."):
        out = predict_image(model, img, preprocess)
    st.session_state.result = out
    emoji = BIN_META[out["predicted_class"]][0]
    c1, c2 = st.columns(2)
    c1.metric("Goes in bin", f"{emoji} {out['predicted_class']}")
    c2.metric("Confidence", f"{out['confidence']:.0%}")
    st.progress(out["confidence"], text="Model confidence")
    if out["manual_review"]:
        st.warning("⚠️ Low confidence - flagged for manual review.")
    else:
        st.success("High-confidence prediction - safe to bin it.")
    c1, c2 = st.columns(2)
    c1.button("← Back to about", on_click=goto, args=(0,))
    c2.button("See confidence breakdown →", on_click=goto, args=(2,),
              type="primary")

# ---------------------------------------------------- Page 3 Confidence ---
else:
    st.subheader("How sure is the AI?")
    out = st.session_state.get("result")
    if out is None:
        st.info("Classify a photo on page 2 first - then come back here.")
        st.button("← Go classify", on_click=goto, args=(1,), type="primary")
        st.stop()
    st.write(f"Top guess: **{BIN_META[out['predicted_class']][0]} "
             f"{out['predicted_class']}** at **{out['confidence']:.0%}**.")
    st.bar_chart(out["all_scores"])
    st.markdown("### What the numbers mean")
    st.write("- **Above 80%**: the AI is sure - bin it.")
    st.write("- **60-80%**: fairly sure - bin it, glance twice.")
    st.write("- **Below 60%**: flagged - let a human decide (manual review).")
    st.button("← Classify another", on_click=goto, args=(1,))
