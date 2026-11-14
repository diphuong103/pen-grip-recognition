"""MediaPipe Tasks HandLandmarker wrapper and lightweight landmark drawing."""
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from .config import ROOT


MODEL_PATH = ROOT / "models" / "hand_landmarker.task"

# MediaPipe's 21-landmark hand topology (wrist, thumb, then four fingers).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


class HandTracker:
    """Detect one hand in IMAGE mode and return normalized xyz landmarks."""

    def __init__(self, model_path=MODEL_PATH):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Hand Landmarker model not found: {model_path}. "
                "Place hand_landmarker.task in the models/ directory."
            )

        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr):
        """Return (21x3 float32 landmarks, handedness) or (None, None)."""
        if frame_bgr is None or frame_bgr.size == 0:
            return None, None
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(image)
        if not result.hand_landmarks:
            return None, None

        landmarks = np.asarray(
            [[point.x, point.y, point.z] for point in result.hand_landmarks[0]],
            dtype=np.float32,
        )
        handedness = result.handedness[0][0].category_name
        return landmarks, handedness

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def draw_hand(frame_bgr, landmarks_list, color=(0, 255, 0)):
    """Draw normalized image landmarks and the hand skeleton onto a BGR frame."""
    if landmarks_list is None:
        return frame_bgr
    height, width = frame_bgr.shape[:2]
    points = np.asarray(landmarks_list)
    if points.shape != (21, 3):
        raise ValueError(f"Expected landmark array with shape (21, 3), got {points.shape}")
    pixels = [
        (int(np.clip(point[0] * width, 0, width - 1)),
         int(np.clip(point[1] * height, 0, height - 1)))
        for point in points
    ]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame_bgr, pixels[start], pixels[end], color, 2, cv2.LINE_AA)
    for x, y in pixels:
        cv2.circle(frame_bgr, (x, y), 4, color, -1, cv2.LINE_AA)
    return frame_bgr
