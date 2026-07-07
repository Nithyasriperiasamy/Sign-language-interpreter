"""
predict.py
----------
Prediction pipeline for the trained ASL CNN model. Loads the model
once and exposes a simple `predict(image)` method that returns the
predicted class name and confidence score.

Can also be run standalone on a single image file for quick testing:

    python predict.py --image path/to/hand.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("OpenCV is required: pip install opencv-python") from exc

import tensorflow as tf

from utils.preprocessing import IMG_SIZE, CLASS_NAMES, preprocess_image

DEFAULT_MODEL_PATH = Path("models/cnn_model.keras")
DEFAULT_CLASS_NAMES_PATH = Path("models/class_names.txt")


class SignLanguagePredictor:
    """Loads the trained CNN and performs inference on hand images."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        class_names_path: Path | str = DEFAULT_CLASS_NAMES_PATH,
        img_size: tuple[int, int] = IMG_SIZE,
    ) -> None:
        """Load the trained model and class names from disk.

        Args:
            model_path: Path to the saved .keras model file.
            class_names_path: Path to the newline-separated class names file
                produced by train.py. Falls back to a default alphabetical
                ordering if the file is not found.
            img_size: Expected model input size (width, height).

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. Train it first with:\n"
                f"    python train.py --data-dir dataset/asl_alphabet_train"
            )

        self.model = tf.keras.models.load_model(model_path)
        self.img_size = img_size

        class_names_path = Path(class_names_path)
        if class_names_path.exists():
            self.class_names = class_names_path.read_text(encoding="utf-8").splitlines()
        else:
            self.class_names = CLASS_NAMES

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        """Predict the ASL class for a given hand image.

        Args:
            image: BGR image (as read by OpenCV) of a cropped hand region.

        Returns:
            A tuple (class_name, confidence) where confidence is in [0, 1].

        Raises:
            ValueError: If the input image is invalid.
        """
        batch = preprocess_image(image, self.img_size)
        predictions = self.model.predict(batch, verbose=0)[0]

        best_index = int(np.argmax(predictions))
        confidence = float(predictions[best_index])
        class_name = (
            self.class_names[best_index] if best_index < len(self.class_names) else "unknown"
        )
        return class_name, confidence

    def predict_top_k(self, image: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        """Return the top-k predicted classes with their confidence scores.

        Args:
            image: BGR image of a cropped hand region.
            k: Number of top predictions to return.

        Returns:
            List of (class_name, confidence) tuples, sorted by confidence descending.
        """
        batch = preprocess_image(image, self.img_size)
        predictions = self.model.predict(batch, verbose=0)[0]

        top_indices = np.argsort(predictions)[::-1][:k]
        return [(self.class_names[i], float(predictions[i])) for i in top_indices]


def _load_predictor_cli(model_path: str, class_names_path: str) -> Optional[SignLanguagePredictor]:
    """Helper for CLI usage; prints a friendly error instead of raising."""
    try:
        return SignLanguagePredictor(model_path=model_path, class_names_path=class_names_path)
    except FileNotFoundError as e:
        print(str(e))
        return None


def main() -> None:
    """CLI entry point for testing predictions on a single image file."""
    parser = argparse.ArgumentParser(description="Run ASL CNN prediction on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to an image file.")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--class-names", type=str, default=str(DEFAULT_CLASS_NAMES_PATH))
    args = parser.parse_args()

    predictor = _load_predictor_cli(args.model, args.class_names)
    if predictor is None:
        return

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to read image: {image_path}")
        return

    class_name, confidence = predictor.predict(image)
    print(f"Prediction: {class_name}  |  Confidence: {confidence * 100:.2f}%")


if __name__ == "__main__":
    main()
