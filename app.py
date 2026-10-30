from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

st.set_page_config(
    page_title="Multi-Organ Histopathology Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

TISSUE_CLASSES = ["Breast", "Colon", "Lung"]
LUNG_CLASSES = ["Adenocarcinoma", "Normal Lung Tissue", "Squamous Cell Carcinoma"]
COLON_CLASSES = ["Adenocarcinoma", "Normal Colon Tissue"]
BREAST_CLASSES = ["Benign", "In Situ Carcinoma", "Invasive Carcinoma", "Normal Breast Tissue"]

st.markdown("""
<style>
    .stApp { background: #ffffff; color: #153958; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stAppViewContainer"] .block-container {
        max-width: 1350px; padding-top: 1.2rem; padding-bottom: 2rem;
    }
    h1, h2, h3 { font-family: Arial, sans-serif; }
    .hero-title {
        color: #78b9e5; font-size: clamp(1.9rem, 3.1vw, 3.1rem);
        font-weight: 800; letter-spacing: .035em; line-height: 1.14;
        text-align: center; margin: 0 0 .55rem;
    }
    .hero-subtitle {
        text-align: center; font-weight: 750; font-size: clamp(.96rem, 1.6vw, 1.35rem);
        color: #153958; letter-spacing: .015em; margin-bottom: 2rem;
    }
    .organ-rail { background: #e8f5ff; border-radius: 28px; padding: 1rem .55rem; }
    .organ-item { text-align:center; margin:.8rem 0 1.65rem; }
    .organ-circle {
        width: 112px; height: 112px; border-radius: 50%; background: white;
        border: 5px solid #79bee9; margin: 0 auto .6rem; display:flex;
        align-items:center; justify-content:center; font-size: 3.3rem;
        box-shadow: 0 3px 12px rgba(30,85,128,.08);
    }
    .organ-circle svg { width: 76px; height: 76px; display:block; }
    .organ-name { color: #23567f; font-size: 1rem; font-weight: 750; }
    .panel-title { font-size: 1.28rem; font-weight: 800; color: #233e59; margin: .15rem 0 1rem; }
    .preview-label { color: #7190a9; font-size: .86rem; margin-bottom: .45rem; }
    .preview-placeholder {
        min-height: 255px; border: 2px dashed #bcd4e7; border-radius: 13px;
        background: #f5faff; display:flex; align-items:center; justify-content:center;
        color:#97adc1; font-size: 1.3rem;
    }
    .result-box {
        background: #f3f8fc; border: 1px solid #d9e8f3; border-radius: 15px;
        padding: 1.2rem 1.3rem; margin-bottom: 1.05rem; min-height: 116px;
    }
    .result-kicker { color: #55738d; font-size: .85rem; font-weight: 700; margin-bottom: .7rem; }
    .result-value { color:#173c60; font-size:1.35rem; font-weight: 800; overflow-wrap: anywhere; }
    .result-empty { color:#9cabb7; font-size:1.05rem; font-weight:500; }
    .disclaimer {
        background: #f1f4f7; border-radius: 14px; padding: 1rem 1.2rem;
        color: #47576a; font-size: .88rem; line-height: 1.5; margin-top:1.65rem;
    }
    div[data-testid="stFileUploader"] { border-radius: 13px; }
    div[data-testid="stFileUploader"] section { background: #f8fbff; border-color: #b5cfdf; }
    div[data-testid="stFileUploader"] button { background: #172d43; color: white; border-radius: 8px; border: none; }
    div[data-testid="stFileUploader"] button:hover { background:#285477; color:white; }
    @media (max-width: 768px) {
        .organ-circle { width: 74px; height: 74px; font-size: 2.1rem; }
        .organ-circle svg { width: 52px; height: 52px; }
        .organ-item { margin: .25rem 0 .65rem; }
        .organ-rail { padding: .65rem; }
        .preview-placeholder { min-height: 150px; }
    }
</style>
""", unsafe_allow_html=True)


def result_card(title: str, value: str | None) -> None:
    display = (f'<div class="result-value">{value}</div>' if value else
               '<div class="result-empty">Waiting for image analysis</div>')
    st.markdown(
        f'<div class="result-box"><div class="result-kicker">{title}</div>{display}</div>',
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_models():
    paths = {
        "tissue": MODEL_DIR / "tissue_mobilenetv2_final.keras",
        "lung": MODEL_DIR / "lung_cnn_baseline.keras",
        "colon": MODEL_DIR / "colon_cnn_baseline.keras",
        "breast": MODEL_DIR / "breast_mobilenetv2_final.keras",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing model file(s):\n" + "\n".join(missing))
    return {name: tf.keras.models.load_model(path) for name, path in paths.items()}


def classify(image: Image.Image, models):
    # Original preprocessing maintained: resize 224x224, RGB float32, no scaling.
    # If training used a different preprocessing routine, match it exactly here.
    array = np.asarray(image.resize((224, 224)), dtype=np.float32)
    batch = np.expand_dims(array, axis=0)
    tissue_pred = models["tissue"].predict(batch, verbose=0)
    tissue_name = TISSUE_CLASSES[int(np.argmax(tissue_pred[0]))]

    if tissue_name == "Lung":
        p = models["lung"].predict(batch, verbose=0)
        histology = LUNG_CLASSES[int(np.argmax(p[0]))]
    elif tissue_name == "Colon":
        p = models["colon"].predict(batch, verbose=0)
        # Keep the exact original binary threshold and class ordering.
        probability = float(p[0][0])
        histology = COLON_CLASSES[1 if probability >= 0.5 else 0]
    else:
        p = models["breast"].predict(batch, verbose=0)
        histology = BREAST_CLASSES[int(np.argmax(p[0]))]
    return tissue_name, histology


st.markdown('<div class="hero-title">MULTI-ORGAN HISTOPATHOLOGY CLASSIFICATION</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">DEEP LEARNING-BASED ANALYSIS OF BREAST, COLON, AND LUNG TISSUE IMAGES</div>', unsafe_allow_html=True)

rail, workspace = st.columns([1.2, 5.3], gap="large")
with rail:
    st.markdown("""
    <div class="organ-rail">
      <div class="organ-item"><div class="organ-circle"><svg viewBox="0 0 100 100" role="img" aria-label="Stylized breast anatomy" xmlns="http://www.w3.org/2000/svg">
        <path d="M48 10 C24 12 13 31 17 51 C20 73 38 86 55 87 C76 88 90 73 88 51 C86 30 70 11 48 10Z" fill="#f6b4c4" stroke="#c95c83" stroke-width="3"/>
        <path d="M47 18 C26 27 25 57 39 77 M53 17 C72 28 77 50 67 76" fill="none" stroke="#ec83a2" stroke-width="3" stroke-linecap="round"/>
        <path d="M48 29 C34 34 35 48 46 57 C53 63 67 59 69 47 C71 35 58 27 48 29Z" fill="#ffe1e6" stroke="#cb7192" stroke-width="2"/>
        <circle cx="51" cy="45" r="5" fill="#bb4f77"/>
        <path d="M50 51 L50 67 M50 61 L40 70 M50 61 L60 71" stroke="#c95c83" stroke-width="2" stroke-linecap="round"/>
      </svg></div><div class="organ-name">BREAST</div></div>
      <div class="organ-item"><div class="organ-circle">🫁</div><div class="organ-name">LUNG</div></div>
      <div class="organ-item"><div class="organ-circle"><svg viewBox="0 0 100 100" role="img" aria-label="Stylized large intestine" xmlns="http://www.w3.org/2000/svg">
        <path d="M23 25 C31 15 70 15 78 25 L79 67 C79 78 68 79 60 83 L56 91" fill="none" stroke="#b85e81" stroke-width="17" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M23 25 C31 15 70 15 78 25 L79 67 C79 78 68 79 60 83 L56 91" fill="none" stroke="#f29bb4" stroke-width="11" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M23 26 L22 65 C22 77 35 78 40 66 L40 42 C40 37 59 37 59 42 L59 63" fill="none" stroke="#b85e81" stroke-width="17" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M23 26 L22 65 C22 77 35 78 40 66 L40 42 C40 37 59 37 59 42 L59 63" fill="none" stroke="#f29bb4" stroke-width="11" stroke-linecap="round" stroke-linejoin="round"/>
      </svg></div><div class="organ-name">COLON</div></div>
    </div>
    """, unsafe_allow_html=True)

with workspace:
    preview_col, result_col = st.columns([1.15, 1], gap="large")
    with preview_col:
        st.markdown('<div class="panel-title">UPLOAD HISTOPATHOLOGY IMAGE</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a histopathology image",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            label_visibility="collapsed",
        )
        st.markdown('<div class="preview-label">IMAGE PREVIEW</div>', unsafe_allow_html=True)
        image = None
        if uploaded_file is None:
            st.markdown('<div class="preview-placeholder">Your image will appear here</div>', unsafe_allow_html=True)
        else:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded histopathology image", use_container_width=True)
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("Unable to read this image. Please upload a valid image file.")

    tissue_name = None
    histology = None
    if image is not None:
        try:
            with st.spinner("Analyzing histopathology image…"):
                models = load_models()
                tissue_name, histology = classify(image, models)
        except Exception as exc:
            st.error("Prediction could not be completed. Verify that all four model files are present and compatible with this environment.")
            with st.expander("Technical details"):
                st.exception(exc)

    with result_col:
        st.markdown('<div class="panel-title">CLASSIFICATION RESULTS</div>', unsafe_allow_html=True)
        result_card("TISSUE IDENTIFICATION", tissue_name)
        result_card("HISTOLOGICAL CLASSIFICATION", histology)
        if tissue_name is not None:
            st.caption("AI model output only — not a diagnosis. These models have not been validated for clinical use.")

    st.markdown("""
    <div class="disclaimer">
       ⚠️ <strong>Educational and research demonstration only.</strong>
       This application is not intended for medical diagnosis or clinical decision-making.
       Upload only supported breast, colon, or lung histopathology images. Unrelated images
       or unfamiliar acquisition conditions may still produce predictions, which can be unreliable.
    </div>
    """, unsafe_allow_html=True)
