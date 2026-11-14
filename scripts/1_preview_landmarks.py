"""Preview a source video with detected MediaPipe hand landmarks."""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import TARGET_HAND
from src.hand_tracker import HandTracker, draw_hand

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import TARGET_HAND


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Path to a source video")
    args = parser.parse_args()
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    wrong_hand_frames = 0
    try:
        with HandTracker() as tracker:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                landmarks, handedness = tracker.detect(frame)
                if landmarks is not None and handedness.lower() == TARGET_HAND.lower():
                    draw_hand(frame, landmarks)
                elif landmarks is not None:
                    wrong_hand_frames += 1
                    cv2.putText(frame, f"WRONG HAND: Use {TARGET_HAND.upper()} hand",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Wrong-hand frames skipped: {wrong_hand_frames}",
                            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("Pen grip preview | press q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
