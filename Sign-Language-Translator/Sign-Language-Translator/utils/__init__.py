"""
Utility package for the CNN-Based Sign Language Translator.

Modules:
    preprocessing   - image preprocessing helpers shared by training/inference
    hand_detector   - MediaPipe-based hand detection and cropping
    tts             - text-to-speech helper for the Streamlit app
"""

from .preprocessing import preprocess_image, IMG_SIZE, CLASS_NAMES
from .hand_detector import HandDetector
from .tts import speak_text

__all__ = [
    "preprocess_image",
    "IMG_SIZE",
    "CLASS_NAMES",
    "HandDetector",
    "speak_text",
]
