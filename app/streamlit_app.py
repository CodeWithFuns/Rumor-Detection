import streamlit as st
import joblib
import os
from pathlib import Path

# --- Determine repo root dynamically ---
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

st.set_page_config(page_title="Rumor Detector", page_icon="📰", layout="centered")

st.title("📰 Rumor Detector")
st.write("Type a sentence/headline and I'll try to guess if it's **Rumor** or **Real**.")

# --- Auto-pick latest model ---
def pick_latest_model(dir_path=None):
    dir_path = Path(dir_path or PROJECT_ROOT / "models")
    try:
        paths = sorted(dir_path.glob("*.joblib"),
                       key=lambda p: p.stat().st_mtime,
                       reverse=True)
        return str(paths[0]) if paths else ""
    except Exception:
        return ""

def resolve_model_path(input_path: str) -> str:
    p = Path(input_path)
    if p.is_absolute() and p.exists():
        return str(p)
    candidates = [
        Path.cwd() / p,                    # where Streamlit was launched
        APP_DIR / p,                        # relative to app/
        PROJECT_ROOT / p,                   # relative to repo root
        PROJECT_ROOT / "models" / p.name,   # allow just the filename
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(p)  # will fail later with a clear error

# --- Default model selection ---
default_model = pick_latest_model()
model_path_raw = st.text_input("Path to trained model (.joblib)", value=default_model)
model_path = resolve_model_path(model_path_raw)

st.caption(f"Working directory: {Path.cwd()}")
st.caption(f"Resolved model path: {model_path}")

if not os.path.exists(model_path):
    st.error(f"Model not found: {model_path}")
    st.stop()

# --- User text input ---
text = st.text_area("Your text", "Breaking: Government bans sleep on weekdays")

# --- Uncertainty threshold slider ---
threshold = st.slider("Uncertainty threshold (lower = more confident)", 0.50, 0.90, 0.65, 0.01)

# --- Prediction button ---
if st.button("Predict"):
    pipe = joblib.load(model_path)
    pred = pipe.predict([text])[0]

    proba = None
    if hasattr(pipe, "predict_proba"):
        try:
            proba = max(pipe.predict_proba([text])[0])
        except Exception:
            proba = None

    if proba is not None and proba < threshold:
        st.warning(f"Unsure (confidence ~ {proba:.2f}). Try more context or rephrase.")
    else:
        st.success(f"Prediction: **{pred}**" + (f"  (confidence ~ {proba:.2f})" if proba is not None else ""))
