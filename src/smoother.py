"""Temporal prediction smoothing for webcam/video streams."""
from collections import Counter, deque

from .config import (
    INSTANT_ALERT_THRESHOLD,
    NO_PEN_INDEX,
    VOTING_THRESHOLD,
    WINDOW_SIZE,
)


class TemporalSmoother:
    """Smooth labels over a sliding window, with an immediate no-pen alert."""

    def __init__(self, window_size=WINDOW_SIZE, voting_threshold=VOTING_THRESHOLD,
                 no_pen_index=NO_PEN_INDEX,
                 instant_alert_threshold=INSTANT_ALERT_THRESHOLD):
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if not 0.0 <= voting_threshold < 1.0:
            raise ValueError("voting_threshold must be in [0, 1)")
        self.history = deque(maxlen=window_size)
        self.voting_threshold = float(voting_threshold)
        self.no_pen_index = int(no_pen_index)
        self.instant_alert_threshold = float(instant_alert_threshold)
        self.stable_label = None

    def update(self, predicted_label, probabilities):
        """Return (stable label/index or None, confidence).

        A confident NO_PEN prediction clears the voting history and is returned
        immediately. `probabilities` is the model's probability vector.
        """
        scores = list(probabilities)
        predicted_label = int(predicted_label)
        confidence = float(scores[predicted_label])
        if (predicted_label == self.no_pen_index
                and confidence >= self.instant_alert_threshold):
            self.history.clear()
            self.stable_label = self.no_pen_index
            return self.no_pen_index, confidence

        self.history.append((predicted_label, confidence))
        winner, votes = Counter(label for label, _ in self.history).most_common(1)[0]
        share = votes / len(self.history)
        winner_confidence = sum(
            score for label, score in self.history if label == winner
        ) / votes
        if share > self.voting_threshold:
            self.stable_label = winner
        else:
            self.stable_label = None
        return self.stable_label, winner_confidence

    def reset(self):
        """Clear queued predictions and the current stable label."""
        self.history.clear()
        self.stable_label = None
