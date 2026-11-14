"""Run pen grip classification on a live webcam feed."""
import argparse
import sys
from pathlib import Path

import cv2
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (CONFIDENCE_THRESHOLD, LABELS, MODELS_DIR,
                        NO_PEN_INDEX, TARGET_HAND, VOTING_THRESHOLD, WINDOW_SIZE)
from src.features import extract_features
from src.hand_tracker import HandTracker, draw_hand
from src.smoother import TemporalSmoother


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    model_path = MODELS_DIR / "pen_grip_mlp.pkl"
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}; run scripts/3_train_model.py first")

    model = joblib.load(model_path)
    smoother = TemporalSmoother(WINDOW_SIZE, VOTING_THRESHOLD)
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")
    try:
        with HandTracker() as tracker:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                landmarks, side = tracker.detect(frame)
                text = "No hand detected"
                color = (40, 220, 80)
                if landmarks is not None:
                    draw_hand(frame, landmarks)
                    if side.lower() != TARGET_HAND.lower():
                        smoother.reset()
                        text = f"WRONG HAND: Use {TARGET_HAND.upper()} hand"
                        color = (0, 0, 255)
                    else:
                        vector = extract_features(landmarks, side).reshape(1, -1)
                        probabilities = model.predict_proba(vector)[0]
                        index = int(probabilities.argmax())
                        confidence = float(probabilities[index])
                        if confidence < CONFIDENCE_THRESHOLD:
                            smoother.reset()
                            text = f"Uncertain ({confidence:.0%})"
                        else:
                            label, score = smoother.update(index, probabilities)
                            if label == NO_PEN_INDEX:
                                text = "NO PEN DETECTED"
                                color = (0, 0, 255)
                            elif label is None:
                                text = f"Stabilizing ({score:.0%})"
                            else:
                                text = f"{LABELS[label]} ({score:.0%})"
                else:
                    smoother.reset()
                cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, color, 2, cv2.LINE_AA)
                cv2.imshow("Pen grip recognition | press q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
