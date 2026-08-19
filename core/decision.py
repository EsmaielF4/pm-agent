"""
Decision layer - the "Decide" stage, i.e. the actual agent reasoning.

This is deliberately separated from the Predictor: the model only says
"how likely is failure". The DecisionPolicy is where your ideas live -
turning a probability (and any other context) into an action: ignore,
watch, schedule maintenance, escalate. This is the layer to extend
first as you add ideas, since it doesn't require retraining anything.
"""

from abc import ABC, abstractmethod

from core.registry import register


class DecisionPolicy(ABC):
    @abstractmethod
    def decide(self, row: dict, prediction: dict) -> dict:
        """row: original raw/feature data for this item (dict).
        prediction: output from the Predictor for this item.
        Returns a decision dict for the Act stage."""
        raise NotImplementedError


@register("decision_policy", "threshold_policy")
class ThresholdDecisionPolicy(DecisionPolicy):
    """Simple, transparent baseline: three probability bands map to three
    actions. Easy to explain in a demo, and easy to replace with
    something smarter later (cost-based decisions, per-equipment
    thresholds, multi-model voting, a policy that also factors in how
    long until the next scheduled maintenance window, etc.) - just
    register a new DecisionPolicy and point config at it."""

    def __init__(self, low: float = 0.3, high: float = 0.7):
        self._low = low
        self._high = high

    def decide(self, row: dict, prediction: dict) -> dict:
        p = prediction["failure_probability"]
        if p >= self._high:
            action, urgency = "SCHEDULE_MAINTENANCE", "high"
            reason = f"Failure probability {p:.0%} exceeds high threshold ({self._high:.0%})"
        elif p >= self._low:
            action, urgency = "FLAG_FOR_INSPECTION", "medium"
            reason = f"Failure probability {p:.0%} in watch band ({self._low:.0%}-{self._high:.0%})"
        else:
            action, urgency = "MONITOR", "low"
            reason = f"Failure probability {p:.0%} below watch threshold ({self._low:.0%})"

        row_id = row.get("udi", row.get("timestamp", row.get("_row_id")))
        return {
            "udi": row_id,
            "action": action,
            "urgency": urgency,
            "reason": reason,
            "failure_probability": p,
        }
