"""Feature extraction shared by dataset creation and live inference."""
import numpy as np


def extract_features(hand_landmarks, handedness="Right"):
    """Return 67 float32 features from a MediaPipe hand.

    The 21 xyz landmarks are wrist-centered, palm-width normalized and rotated
    so the wrist-to-middle-MCP vector is vertical. Before rotation, the wrist
    axis angle's sine and cosine are saved to preserve wrist orientation cues.
    """
    if isinstance(hand_landmarks, np.ndarray):
        points = np.asarray(hand_landmarks, dtype=np.float32)
    else:
        points = np.asarray(
            [[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32
        )
    if points.shape != (21, 3):
        raise ValueError(f"Expected 21 xyz landmarks, got shape {points.shape}")

    points -= points[0].copy()
    palm_width = float(np.linalg.norm(points[5, :2] - points[17, :2]))
    if palm_width > 1e-6:
        points /= palm_width

    # Preserve wrist flexion/orientation before normalizing the hand rotation.
    wrist_angle = np.arctan2(points[9, 1], points[9, 0])
    wrist_orientation = np.asarray(
        [np.sin(wrist_angle), np.cos(wrist_angle)], dtype=np.float32
    )

    angle = np.arctan2(points[9, 0], -points[9, 1])
    c, s = np.cos(-angle), np.sin(-angle)
    points[:, :2] = points[:, :2] @ np.asarray([[c, s], [-s, c]], dtype=np.float32)
    is_left = str(handedness).strip().lower() == "left"
    handedness_flags = np.asarray([0.0, 1.0] if is_left else [1.0, 0.0], dtype=np.float32)
    distances = np.asarray(
        [np.linalg.norm(points[4] - points[8]), np.linalg.norm(points[8] - points[12])],
        dtype=np.float32,
    )
    features = np.concatenate((points.reshape(-1), handedness_flags, distances, wrist_orientation))
    if features.shape != (69,):
        raise RuntimeError(f"Expected 69 features, got {features.size}")
    return features
