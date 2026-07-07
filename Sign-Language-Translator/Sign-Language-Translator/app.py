"""
app.py
------
Streamlit web application for the CNN-Based Sign Language Translator.

Run locally with:
    streamlit run app.py

Deployable as-is on Hugging Face Spaces (Streamlit SDK).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

try:
    import cv2
except ImportError:
    st.error("OpenCV is not installed. Run: pip install opencv-python-headless")
    st.stop()

from predict import SignLanguagePredictor, DEFAULT_MODEL_PATH, DEFAULT_CLASS_NAMES_PATH
from utils.hand_detector import HandDetector
from utils.tts import speak_text

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="Sign Language Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Styling — a signal-forward, dark "vision system" theme.
# Deep slate background, teal/violet duotone accents (nod to CV heatmaps),
# monospace type for data readouts, humanist sans for body copy.
# ----------------------------------------------------------------------

CUSTOM_CSS = """
<style>
:root {
    --bg-deep:      #0b0f14;
    --bg-panel:     #121821;
    --line:         #232c38;
    --teal:         #2dd4bf;
    --violet:       #a78bfa;
    --text-main:    #e6edf3;
    --text-dim:     #8b98a8;
}

.stApp {
    background: var(--bg-deep);
    color: var(--text-main);
}

h1, h2, h3 {
    font-family: 'Georgia', serif;
    letter-spacing: -0.01em;
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, var(--teal), var(--violet));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.hero-subtitle {
    color: var(--text-dim);
    font-size: 1rem;
    margin-top: 0.2rem;
}

.panel {
    background: var(--bg-panel);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}

.letter-readout {
    font-family: 'Courier New', monospace;
    font-size: 4rem;
    font-weight: 700;
    color: var(--teal);
    text-align: center;
    line-height: 1;
}

.confidence-text {
    font-family: 'Courier New', monospace;
    text-align: center;
    color: var(--text-dim);
    font-size: 0.95rem;
}

.sentence-box {
    font-family: 'Courier New', monospace;
    font-size: 1.3rem;
    color: var(--text-main);
    background: var(--bg-deep);
    border: 1px dashed var(--line);
    border-radius: 8px;
    padding: 1rem;
    min-height: 3rem;
    word-wrap: break-word;
}

div[data-testid="stMetricValue"] {
    color: var(--teal);
}

.stButton > button {
    background: var(--bg-panel);
    border: 1px solid var(--line);
    color: var(--text-main);
    border-radius: 8px;
    transition: border-color 0.15s ease;
}
.stButton > button:hover {
    border-color: var(--teal);
    color: var(--teal);
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Session state initialization
# ----------------------------------------------------------------------

if "sentence" not in st.session_state:
    st.session_state.sentence = ""
if "history" not in st.session_state:
    st.session_state.history = []  # list of (letter, confidence) tuples
if "current_letter" not in st.session_state:
    st.session_state.current_letter = "-"
if "current_confidence" not in st.session_state:
    st.session_state.current_confidence = 0.0


@st.cache_resource(show_spinner=False)
def load_predictor() -> SignLanguagePredictor | None:
    """Load the trained CNN model once and cache it across reruns.

    Returns:
        A SignLanguagePredictor instance, or None if the model file is missing.
    """
    try:
        return SignLanguagePredictor(
            model_path=DEFAULT_MODEL_PATH, class_names_path=DEFAULT_CLASS_NAMES_PATH
        )
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def load_hand_detector() -> HandDetector:
    """Create and cache a single MediaPipe HandDetector instance."""
    return HandDetector(max_num_hands=1, padding=25)


def add_letter(letter: str) -> None:
    """Append a letter (or space/delete action) to the sentence in session state."""
    if letter == "space":
        st.session_state.sentence += " "
    elif letter == "del":
        st.session_state.sentence = st.session_state.sentence[:-1]
    elif letter == "nothing":
        pass
    else:
        st.session_state.sentence += letter


def process_frame(frame: np.ndarray, detector: HandDetector, predictor: SignLanguagePredictor):
    """Run hand detection + CNN prediction on a single frame.

    Args:
        frame: BGR frame from webcam or uploaded image.
        detector: HandDetector instance.
        predictor: SignLanguagePredictor instance.

    Returns:
        Tuple (annotated_frame, cropped_hand_or_None, class_name_or_None, confidence).
    """
    annotated = detector.draw_bbox(frame)
    cropped = detector.crop_hand(frame)

    if cropped is None or cropped.size == 0:
        return annotated, None, None, 0.0

    class_name, confidence = predictor.predict(cropped)
    return annotated, cropped, class_name, confidence


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🤟 Model Info")
    model_exists = Path(DEFAULT_MODEL_PATH).exists()
    if model_exists:
        st.success("CNN model loaded")
        st.caption(f"Path: `{DEFAULT_MODEL_PATH}`")
    else:
        st.error("Model not found")
        st.caption(
            "Train it first with:\n\n```\npython train.py --data-dir dataset/asl_alphabet_train\n```"
        )

    st.markdown("---")
    st.markdown("## 📋 Instructions")
    st.markdown(
        """
        1. Choose **Webcam** or **Upload Image** below.
        2. Show a single hand making an ASL letter.
        3. Wait for a stable, confident prediction.
        4. Click **Add Letter** to append it to your sentence.
        5. Use **Space** / **Delete** / **Clear** to edit.
        6. Click **Speak Sentence** to hear it aloud.
        """
    )

    st.markdown("---")
    st.markdown("## ⌨️ Keyboard Shortcuts")
    st.caption(
        "Streamlit doesn't support global key bindings natively, "
        "so shortcuts are mapped to on-screen buttons below. "
        "For power users: focus a button and press **Enter** to trigger it."
    )

    st.markdown("---")
    dark_mode_note = st.checkbox("Dark mode (default)", value=True, disabled=True)
    st.caption("This app is designed dark-mode-first for best contrast on camera feeds.")

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------

st.markdown('<p class="hero-title">Sign Language Translator</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Real-time ASL alphabet recognition powered by a custom CNN + MediaPipe Hands</p>',
    unsafe_allow_html=True,
)
st.write("")

predictor = load_predictor()
detector = load_hand_detector()

if predictor is None:
    st.warning(
        "⚠️ No trained model found at `models/cnn_model.keras`. "
        "Please run `python train.py` first, or upload a pretrained model to the `models/` folder."
    )

# ----------------------------------------------------------------------
# Main layout: input column + live prediction/sentence column
# ----------------------------------------------------------------------

input_col, output_col = st.columns([1.2, 1])

with input_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 📷 Input")
    input_mode = st.radio("Choose input source", ["Webcam", "Upload Image"], horizontal=True)

    frame_bgr: np.ndarray | None = None

    if input_mode == "Webcam":
        camera_image = st.camera_input("Show your hand sign to the camera")
        if camera_image is not None:
            file_bytes = np.frombuffer(camera_image.getvalue(), dtype=np.uint8)
            frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
            frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame_bgr is not None and predictor is not None:
        annotated_frame, cropped_hand, class_name, confidence = process_frame(
            frame_bgr, detector, predictor
        )
        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="Detected hand region", use_container_width=True)

        if cropped_hand is None:
            st.info("No hand detected.")
            st.session_state.current_letter = "-"
            st.session_state.current_confidence = 0.0
        else:
            st.session_state.current_letter = class_name
            st.session_state.current_confidence = confidence
            st.session_state.history.append((class_name, confidence))
            st.session_state.history = st.session_state.history[-10:]  # keep last 10
    st.markdown("</div>", unsafe_allow_html=True)

with output_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 🔮 Live Prediction")
    st.markdown(
        f'<div class="letter-readout">{st.session_state.current_letter}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="confidence-text">confidence: {st.session_state.current_confidence * 100:.1f}%</div>',
        unsafe_allow_html=True,
    )
    st.progress(min(max(st.session_state.current_confidence, 0.0), 1.0))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 🧠 Recent Predictions")
    if st.session_state.history:
        for letter, conf in reversed(st.session_state.history):
            st.caption(f"`{letter}` — {conf * 100:.1f}%")
    else:
        st.caption("No predictions yet.")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Sentence builder
# ----------------------------------------------------------------------

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown("### 📝 Sentence Builder")
st.markdown(f'<div class="sentence-box">{st.session_state.sentence or "&nbsp;"}</div>', unsafe_allow_html=True)
st.write("")

btn_cols = st.columns(5)

with btn_cols[0]:
    if st.button("➕ Add Letter", use_container_width=True):
        if st.session_state.current_letter not in ("-", None):
            add_letter(st.session_state.current_letter)

with btn_cols[1]:
    if st.button("⌫ Delete Letter", use_container_width=True):
        st.session_state.sentence = st.session_state.sentence[:-1]

with btn_cols[2]:
    if st.button("␣ Space", use_container_width=True):
        st.session_state.sentence += " "

with btn_cols[3]:
    if st.button("🗑️ Clear Sentence", use_container_width=True):
        st.session_state.sentence = ""

with btn_cols[4]:
    speak_clicked = st.button("🔊 Speak Sentence", use_container_width=True)

if speak_clicked:
    audio_bytes = speak_text(st.session_state.sentence)
    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3")
    else:
        st.warning("Could not generate speech (empty sentence or no internet access).")

st.download_button(
    "⬇️ Download Sentence as TXT",
    data=st.session_state.sentence,
    file_name="translated_sentence.txt",
    mime="text/plain",
    use_container_width=False,
)

st.markdown("</div>", unsafe_allow_html=True)

st.caption(
    "Built with TensorFlow/Keras, OpenCV, MediaPipe Hands, and Streamlit. "
    "For best results, use good lighting and a plain background."
)
