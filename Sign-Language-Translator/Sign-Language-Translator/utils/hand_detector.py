"""
hand_detector.py
----------------
Wraps MediaPipe Hands to provide a simple, reusable interface for
detecting a single hand in a frame, drawing landmarks, and cropping
the hand region (with padding) for CNN inference.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import cv2
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "OpenCV and MediaPipe are required. Install with: "
        "pip install opencv-python mediapipe"
    ) from exc

from .preprocessing import add_padding_to_bbox


class HandDetector:
    """Detects a single hand in a video frame using MediaPipe Hands."""

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        padding: int = 20,
    ) -> None:
        """Initialize the MediaPipe Hands model.

        Args:
            max_num_hands: Maximum number of hands to detect (we only need one).
            min_detection_confidence: Minimum confidence for initial detection.
            min_tracking_confidence: Minimum confidence to keep tracking a hand.
            padding: Extra pixels added around the detected hand's bounding box.
        """
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self.padding = padding

        self.hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def find_hand_bbox(self, frame: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        """Locate the bounding box of a hand in the given frame.

        Args:
            frame: BGR image (as returned by cv2.VideoCapture / cv2.imread).

        Returns:
            (x1, y1, x2, y2) padded bounding box in pixel coordinates,
            or None if no hand was detected.
        """
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        x_coords = [lm.x * frame_width for lm in hand_landmarks.landmark]
        y_coords = [lm.y * frame_height for lm in hand_landmarks.landmark]

        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        w, h = x_max - x_min, y_max - y_min
        x1, y1, x2, y2 = add_padding_to_bbox(
            x_min, y_min, w, h, frame_width, frame_height, self.padding
        )
        return x1, y1, x2, y2

    def crop_hand(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect and crop the hand region from a frame.

        Args:
            frame: BGR image.

        Returns:
            Cropped hand image (BGR), or None if no hand was found.
        """
        bbox = self.find_hand_bbox(frame)
        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2].copy()

    def draw_bbox(self, frame: np.ndarray, color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw the detected hand's bounding box onto a copy of the frame.

        Args:
            frame: BGR image.
            color: BGR color for the rectangle.

        Returns:
            A new frame with the bounding box drawn (or the original frame
            unmodified if no hand is detected).
        """
        output = frame.copy()
        bbox = self.find_hand_bbox(frame)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        return output

    def close(self) -> None:
        """Release MediaPipe resources. Call when done with the detector."""
        self.hands.close()

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
