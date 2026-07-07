# 🤟 CNN-Based Sign Language Translator

A real-time web application that recognizes American Sign Language (ASL)
alphabet hand gestures using a custom Convolutional Neural Network (CNN)
and translates them into text — with optional text-to-speech playback.

Built with **TensorFlow/Keras**, **OpenCV**, **MediaPipe Hands**, and
**Streamlit**, and deployable directly on **Hugging Face Spaces**.

---

## ✨ Features

- 🖐️ Real-time hand detection via MediaPipe Hands
- 🧠 Custom CNN trained on the ASL Alphabet dataset (29 classes)
- 📷 Webcam capture **and** image upload support
- 🔤 Live letter prediction with confidence score
- 📝 Sentence builder with Add / Delete / Space / Clear controls
- 🔊 Text-to-speech playback of the built sentence
- 📊 Prediction history panel
- ⬇️ Download the translated sentence as a `.txt` file
- 🌙 Dark-mode-first, responsive UI

---

## 🏗️ Architecture

**CNN (Keras Sequential API):**

```
Input (64x64x3)
 → Conv2D(32) → ReLU → MaxPooling2D
 → Conv2D(64) → ReLU → MaxPooling2D
 → Conv2D(128) → ReLU
 → Flatten
 → Dense(256) → ReLU → Dropout(0.5)
 → Dense(29) → Softmax
```

**Training pipeline:**

- `ImageDataGenerator` with rotation, shift, shear, zoom, and brightness augmentation
- `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau` callbacks
- Automatic accuracy/loss plots and a confusion matrix saved to `models/`

**Inference pipeline:**

- MediaPipe Hands detects and crops the hand region (with padding)
- The crop is resized to 64×64, normalized, and fed to the CNN
- Returns the predicted letter + confidence score

---

## 📦 Dataset

This project uses the **ASL Alphabet** dataset (Kaggle):
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

29 classes: `A`–`Z`, `space`, `del`, `nothing`.

See [`dataset/README.md`](dataset/README.md) for full download and setup
instructions. In short:

```bash
pip install kaggle
kaggle datasets download -d grassknoted/asl-alphabet
unzip asl-alphabet.zip -d dataset/
```

Expected structure:

```
dataset/
    asl_alphabet_train/
        A/  B/  C/  ...  Z/
        space/
        del/
        nothing/
```

---

## 📁 Folder Structure

```
Sign-Language-Translator/
├── dataset/
│   └── README.md            # Dataset download & setup instructions
├── models/                  # Trained model + plots saved here
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py     # Shared image preprocessing
│   ├── hand_detector.py     # MediaPipe hand detection & cropping
│   └── tts.py                # Text-to-speech helper
├── assets/                  # Screenshots / static assets
├── train.py                  # CNN training script
├── predict.py                 # Prediction pipeline / CLI
├── app.py                     # Streamlit web application
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Installation

```bash
git clone https://github.com/<your-username>/Sign-Language-Translator.git
cd Sign-Language-Translator

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## 🏋️ Training

1. Set up the dataset as described above.
2. Run:

```bash
python train.py --data-dir dataset/asl_alphabet_train --epochs 30
```

This saves:

- `models/cnn_model.keras` — the trained model
- `models/class_names.txt` — class label order
- `models/accuracy_plot.png`, `models/loss_plot.png`
- `models/confusion_matrix.png`

---

## ▶️ Running Locally

```bash
streamlit run app.py
```

Then open the printed local URL (typically `http://localhost:8501`) in your browser.

---

## ☁️ Deployment on Hugging Face Spaces

1. Create a new Space at https://huggingface.co/new-space
   - **SDK:** Streamlit
   - **Hardware:** CPU basic is sufficient
2. Push this repository's contents to the Space's git remote:

```bash
git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
git push space main
```

3. Ensure `app.py` is at the repository root (it is) — Hugging Face Spaces
   auto-detects it as the Streamlit entry point.
4. Upload a trained `models/cnn_model.keras` and `models/class_names.txt`
   to the Space (via the web UI, `git lfs`, or the `huggingface_hub` API),
   since the raw dataset and model are excluded from git by `.gitignore`.
5. The Space will build automatically using `requirements.txt`.

**Space configuration (`README.md` front matter for HF Spaces)**, add this
block to the top of your Space's README if the Space doesn't detect settings
automatically:

```yaml
---
title: Sign Language Translator
emoji: 🤟
colorFrom: teal
colorTo: purple
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---
```

---

## 🔧 Git Commands

```bash
git init
git add .
git commit -m "Initial commit: CNN-based Sign Language Translator"
git branch -M main
git remote add origin https://github.com/<your-username>/Sign-Language-Translator.git
git push -u origin main
```

---

## 🔮 Future Improvements

- Extend to full ASL word/phrase recognition using sequence models (LSTM/Transformer)
- Add two-hand and dynamic-gesture support
- Quantize the model (TFLite) for faster edge/mobile inference
- Add user-specific calibration for lighting/skin tone robustness
- Multi-language text-to-speech and translation

---

## 📸 Screenshots

> _Add screenshots of the running app here._

`assets/screenshot_home.png` — Main interface
`assets/screenshot_prediction.png` — Live prediction in action
`assets/screenshot_sentence.png` — Sentence builder

---

## 📄 License

This project is provided as an educational/portfolio reference implementation.
Check the ASL Alphabet dataset's Kaggle license terms before commercial use.
