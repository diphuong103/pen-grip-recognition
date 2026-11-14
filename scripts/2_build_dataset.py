"""Extract 67 landmark features from raw videos into the training CSV."""
import argparse
import re
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (DATASET_CSV, FEATURE_COUNT, LABELS, RAW_DIR,
                        TARGET_HAND, VIDEO_EXTENSIONS, VIDEO_NAME_PATTERN)
from src.features import extract_features
from src.hand_tracker import HandTracker


def parse_video_name(video_path):
    """Parse u<x>_<y> (e.g. u1_2.mp4) and return canonical user id, or None."""
    match = re.fullmatch(VIDEO_NAME_PATTERN, video_path.stem)
    return f"u{int(match.group('user'))}" if match else None


def iter_videos():
    for label_name in LABELS:
        folder = RAW_DIR / label_name
        if not folder.is_dir():
            print(f"[WARN] Missing label folder: {folder}")
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                yield label_name, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-step", type=int, default=3,
                        help="Process every Nth frame (default: 3)")
    parser.add_argument("--output", type=Path, default=DATASET_CSV)
    args = parser.parse_args()
    if args.frame_step < 1:
        parser.error("--frame-step must be at least 1")

    videos = list(iter_videos())
    if not videos:
        raise SystemExit(f"No supported videos found in {RAW_DIR}")
    rows, invalid_names, unreadable = [], [], []
    wrong_hand_frames = 0
    with HandTracker() as tracker:
        for label_name, path in videos:
            user_id = parse_video_name(path)
            if user_id is None:
                invalid_names.append(path)
                print(f"[SKIP] Invalid filename (expected u<x>_<y>): {path.name}")
                continue

            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                unreadable.append(path)
                print(f"[WARN] Cannot open video: {path}")
                continue
            label_id = LABELS.index(label_name)
            frame_number = 0
            extracted = 0
            video_wrong_hand_frames = 0
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if frame_number % args.frame_step:
                        frame_number += 1
                        continue
                    frame_number += 1
                    landmarks, side = tracker.detect(frame)
                    if landmarks is None:
                        continue
                    if side.lower() != TARGET_HAND.lower():
                        wrong_hand_frames += 1
                        video_wrong_hand_frames += 1
                        continue
                    features = extract_features(landmarks, side)
                    row = {"user_id": user_id, "label": label_id}
                    row.update({f"f{i}": float(value) for i, value in enumerate(features)})
                    rows.append(row)
                    extracted += 1
            finally:
                capture.release()
            print(
                f"[OK] {path.name}: {extracted} feature rows, label={label_name}; "
                f"skipped wrong-hand frames={video_wrong_hand_frames}"
            )

    columns = ["user_id", "label", *[f"f{i}" for i in range(FEATURE_COUNT)]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} rows to {args.output}")
    print(
        f"Skipped invalid names: {len(invalid_names)}; unreadable videos: {len(unreadable)}; "
        f"wrong-hand frames: {wrong_hand_frames}"
    )


if __name__ == "__main__":
    main()
