"""Configuration constants shared across the training and inference pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw_videos"
DATASET_CSV = ROOT / "data" / "processed" / "pen_grip_dataset.csv"
MODELS_DIR = ROOT / "models"

LABELS = (
    "0_Dynamic_Tripod",
    "1_Thumb_Wrap",
    "2_Thumb_Tuck",
    "3_Thumb_Over_Middle",
    "4_Fisted_Grip",
    "5_Grip_Too_Low",
    "6_Grip_Too_High",
    "7_Hooked_Wrist",
    "8_No_Pen",
)
FEATURE_COUNT = 69
TARGET_HAND = "Right"
CONFIDENCE_THRESHOLD = 0.70
WINDOW_SIZE = 15
VOTING_THRESHOLD = 0.75
NO_PEN_INDEX = 8
INSTANT_ALERT_THRESHOLD = 0.90
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
VIDEO_NAME_PATTERN = r"(?i)^u(?P<user>\d+)_(?P<video>\d+)$"
