"""
Adaptive baseline decision policy - replaces fixed round-number
thresholds with two ideas drawn directly from real maintenance/PdM
practice:

1. Dynamic baseline (addresses alarm fatigue): a fixed threshold like
   "70%" is an arbitrary guess. Real PdM systems that survive contact
   with actual maintenance crews instead learn what the model's score
   normally looks like for HEALTHY operation of THIS asset, then flag
   based on how far a new reading deviates from that - in standard-
   deviation units (a z-score) rather than an absolute number picked
   by hand.

2. Trend-based early warning (uses the P-F curve concept): a score
   climbing fast toward the threshold is itself informative, even
   before it crosses the absolute line. Catching that trend earlier
   effectively lengthens the usable P-F interval - more lead time for
   the maintenance team to schedule work during a planned window
   instead of reacting to a hard alarm.
"""

from collections import deque
import statistics

from core.registry import register
from core.decision import DecisionPolicy


@register("decision_policy", "adaptive_baseline_policy")
class AdaptiveBaselinePolicy(DecisionPolicy):
    def __init__(
        self,
        z_medium: float = 2.0,
        z_high: float = 3.5,
        trend_window: int = 5,
        trend_escalate_slope: float = 0.08,
    ):
        self._z_medium = z_medium
        self._z_high = z_high
        self._trend_window = trend_window
        self._trend_escalate_slope = trend_escalate_slope
        self._baseline_mean = 0.0
        self._baseline_std = 1.0
        self._history = deque(maxlen=trend_window)

    def calibrate(self, predictions: list, labels) -> None:
        """Learns the baseline from the model's own predicted
        failure_probability on TRAINING rows belonging to the healthy/
        normal class (whichever label scores lower on average - this
        works regardless of how that class happens to be named)."""
        scores = [p.get("failure_probability") for p in predictions]
        if any(s is None for s in scores):
            return  # this predictor's output isn't probability-shaped; skip safely

        labels = list(labels)
        classes = sorted(set(labels))
        if len(classes) >= 2:
            avg_by_class = {}
            for c in classes:
                class_scores = [s for s, l in zip(scores, labels) if l == c]
                avg_by_class[c] = sum(class_scores) / max(1, len(class_scores))
            healthy_class = min(avg_by_class, key=avg_by_class.get)
            healthy_scores = [s for s, l in zip(scores, labels) if l == healthy_class]
        else:
            healthy_scores = scores

        if len(healthy_scores) >= 2:
            self._baseline_mean = statistics.mean(healthy_scores)
            self._baseline_std = statistics.pstdev(healthy_scores) or 0.01
        else:
            self._baseline_mean, self._baseline_std = 0.0, 1.0

    def decide(self, row: dict, prediction: dict) -> dict:
        p = prediction["failure_probability"]
        z = (p - self._baseline_mean) / self._baseline_std

        self._history.append(p)
        trend_slope = 0.0
        if len(self._history) >= 2:
            trend_slope = (self._history[-1] - self._history[0]) / (len(self._history) - 1)

        early_warning = trend_slope >= self._trend_escalate_slope and z < self._z_high

        if z >= self._z_high:
            action, urgency = "SCHEDULE_MAINTENANCE", "high"
            reason = (
                f"Score {p:.0%} is {z:.1f} std devs above the learned healthy "
                f"baseline ({self._baseline_mean:.0%})"
            )
        elif z >= self._z_medium or early_warning:
            action, urgency = "FLAG_FOR_INSPECTION", "medium"
            if early_warning:
                reason = (
                    f"Score rising fast (avg +{trend_slope:.0%}/reading over the "
                    f"last {len(self._history)}) - early warning ahead of the P-point"
                )
            else:
                reason = f"Score {p:.0%} is {z:.1f} std devs above the healthy baseline"
        else:
            action, urgency = "MONITOR", "low"
            reason = f"Score {p:.0%} is within normal range for this asset ({z:.1f} std devs)"

        row_id = row.get("udi", row.get("timestamp", row.get("_row_id")))
        return {
            "udi": row_id,
            "action": action,
            "urgency": urgency,
            "reason": reason,
            "failure_probability": p,
        }