"""
tts.py
------
Lightweight text-to-speech helper for the Streamlit app.

Uses gTTS (Google Text-to-Speech) to generate an MP3 in memory, which
Streamlit can then play with st.audio(). gTTS is used instead of
pyttsx3 because pyttsx3 relies on system audio drivers that are not
available in headless environments like Hugging Face Spaces.
"""

from __future__ import annotations

import io


def speak_text(text: str, lang: str = "en") -> bytes | None:
    """Convert text to speech and return the resulting MP3 audio as bytes.

    Args:
        text: The sentence to synthesize. Empty/whitespace-only text is ignored.
        lang: Language code for gTTS (default English).

    Returns:
        Raw MP3 bytes suitable for st.audio(), or None if `text` is empty
        or speech synthesis fails (e.g., no internet access).
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return None

    try:
        from gtts import gTTS
    except ImportError:
        # gTTS not installed - fail gracefully rather than crashing the app.
        return None

    try:
        tts = gTTS(text=cleaned_text, lang=lang)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()
    except Exception:
        # Network issues, unsupported language, etc. - fail gracefully.
        return None
