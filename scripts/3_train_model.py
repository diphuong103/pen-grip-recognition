"""Train, evaluate, and export the pen grip MLP model."""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATASET_CSV, FEATURE_COUNT, LABELS, MODELS_DIR


def main():
    if not DATASET_CSV.is_file():
        raise SystemExit(f"Dataset not found at {DATASET_CSV}; run scripts/2_build_dataset.py first")
    data = pd.read_csv(DATASET_CSV)
    feature_columns = [f"f{i}" for i in range(FEATURE_COUNT)]
    required_columns = ["user_id", "label", *feature_columns]
    missing = [name for name in required_columns if name not in data.columns]
    if missing:
        raise SystemExit("Missing dataset columns: " + ", ".join(missing))
    data = data.dropna(subset=required_columns).copy()
    if data.empty or data.user_id.nunique() < 2:
        raise SystemExit("Need usable samples from at least two users for a grouped train/test split")

    X = data[feature_columns].astype("float32")
    y = data.label.astype(int)
    groups = data.user_id.astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    train_labels, test_labels = set(y.iloc[train_idx]), set(y.iloc[test_idx])
    missing_train = sorted(set(range(len(LABELS))) - train_labels)
    if missing_train:
        absent = ", ".join(LABELS[index] for index in missing_train)
        raise SystemExit(f"Training split is missing labels: {absent}. Add more users per label.")
    missing_test = sorted(set(range(len(LABELS))) - test_labels)
    if missing_test:
        print("[WARN] Test split has no samples for: " + ", ".join(LABELS[i] for i in missing_test))

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                      early_stopping=True, random_state=42),
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    predictions = model.predict(X.iloc[test_idx])
    print(f"Train users: {groups.iloc[train_idx].nunique()}; test users: {groups.iloc[test_idx].nunique()}")
    print(classification_report(
        y.iloc[test_idx], predictions, labels=range(len(LABELS)),
        target_names=LABELS, zero_division=0,
    ))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "pen_grip_mlp.pkl")
    display = ConfusionMatrixDisplay.from_predictions(
        y.iloc[test_idx], predictions, labels=range(len(LABELS)),
        display_labels=LABELS, xticks_rotation=0, cmap="Blues",
    )
    ax = display.ax_
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels([str(index) for index in range(len(LABELS))], rotation=0)
    ax.set_xticks([index - 0.5 for index in range(len(LABELS) + 1)], minor=True)
    ax.set_yticks([index - 0.5 for index in range(len(LABELS) + 1)], minor=True)
    ax.grid(which="minor", color="#d3d3d3", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    display.figure_.tight_layout()
    display.figure_.savefig(
        MODELS_DIR / "confusion_matrix.png", dpi=160, bbox_inches="tight"
    )
    plt.close(display.figure_)

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        onnx_model = convert_sklearn(
            model,
            initial_types=[("features", FloatTensorType([None, FEATURE_COUNT]))],
            options={id(model[-1]): {"zipmap": False}},
        )
        (MODELS_DIR / "pen_grip_mlp.onnx").write_bytes(onnx_model.SerializeToString())
    except ImportError:
        print("[WARN] skl2onnx unavailable; skipped ONNX export")
    except Exception as exc:
        print(f"[WARN] ONNX export failed: {exc}")


if __name__ == "__main__":
    main()
