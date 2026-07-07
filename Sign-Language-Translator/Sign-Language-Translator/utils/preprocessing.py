"""
preprocessing.py
----------------
Shared image preprocessing utilities used by both the training pipeline
(train.py) and the inference pipeline (predict.py, app.py).

Keeping preprocessing logic in one place guarantees that images are
prepared identically at train time and at inference time, which is
critical for a CNN to perform correctly.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "OpenCV is required. Install it with: pip install opencv-python"
    ) from exc

# ----------------------------------------------------------------------
# Global constants
# ----------------------------------------------------------------------

# Input size expected by the CNN (width, height). 64x64 keeps the model
# lightweight enough to run in real time on CPU-only Hugging Face Spaces.
IMG_SIZE: tuple[int, int] = (64, 64)

# Class names in the exact order produced by Keras' ImageDataGenerator
# (alphabetical). A-Z, then 'del', 'nothing', 'space' sort after the
# uppercase letters alphabetically when lower-cased, so we keep this
# list explicit and let train.py verify it matches the generator's
# class_indices at training time.
CLASS_NAMES: list[str] = (
    [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["del", "nothing", "space"]
)


def preprocess_image(image: np.ndarray, img_size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Prepare a raw BGR/RGB image (as read by OpenCV) for CNN inference.

    Steps performed:
        1. Convert to RGB (Keras models trained on RGB data expect this order).
        2. Resize to the model's expected input size.
        3. Normalize pixel values to the [0, 1] range.
        4. Add a batch dimension.

    Args:
        image: Input image as a NumPy array (H, W, 3), any color order.
        img_size: Target (width, height) for the model input.

    Returns:
        A 4D NumPy array of shape (1, H, W, 3) ready to feed to model.predict().

    Raises:
        ValueError: If the input image is empty or not a valid 3-channel image.
    """
    if image is None or image.size == 0:
        raise ValueError("Received an empty image for preprocessing.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a 3-channel color image, got shape {image.shape}.")

    # OpenCV loads images as BGR by default; convert to RGB for consistency
    # with ImageDataGenerator (which reads files directly as RGB via PIL).
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(rgb_image, img_size, interpolation=cv2.INTER_AREA)

    normalized = resized.astype("float32") / 255.0

    batched = np.expand_dims(normalized, axis=0)
    return batched


def add_padding_to_bbox(
    x: int, y: int, w: int, h: int, frame_width: int, frame_height: int, padding: int = 20
) -> tuple[int, int, int, int]:
    """Expand a bounding box by `padding` pixels on every side, clamped to frame bounds.

    Args:
        x, y, w, h: Original bounding box (top-left x, y, width, height).
        frame_width, frame_height: Dimensions of the source frame.
        padding: Number of pixels to add on each side.

    Returns:
        A tuple (x1, y1, x2, y2) representing the padded box corners.
    """
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, frame_width)
    y2 = min(y + h + padding, frame_height)
    return x1, y1, x2, y2
