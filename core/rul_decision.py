"""
Decision policy for RUL-based maintenance scheduling (P3).

A third distinct reasoning shape: P2 asks "how likely is failure",
P4 asks "what kind of fault and who handles it", this one asks "how
much time is left, and does that leave enough runway to schedule
maintenance normally or do we need to act now". Thresholds below are
illustrative defaults (same honest caveat as P2's threshold_policy) -
in a real deployment they should be set from the actual lead time
needed to schedule a maintenance crew/parts, not a round number.
"""

from core.registry import register
from core.decision import DecisionPolicy


@register("decision_policy", "rul_maintenance_policy")
class RULMaintenancePolicy(DecisionPolicy):
    def __init__(self, critical_seconds: float = 250, warning_seconds: float = 400):
        self._critical = critical_seconds
        self._warning = warning_seconds

    def decide(self, row: dict, prediction: dict) -> dict:
        rul = prediction["predicted_rul_seconds"]

        if rul <= self._critical:
            action, urgency = "SCHEDULE_MAINTENANCE", "high"
            reason = f"Predicted RUL {rul:.0f}s is below the critical threshold ({self._critical:.0f}s)"
        elif rul <= self._warning:
            action, urgency = "FLAG_FOR_INSPECTION", "medium"
            reason = f"Predicted RUL {rul:.0f}s is in the warning band ({self._critical:.0f}-{self._warning:.0f}s)"
        else:
            action, urgency = "MONITOR", "low"
            reason = f"Predicted RUL {rul:.0f}s is above the warning threshold ({self._warning:.0f}s)"

        # normalized 0-1 risk score, purely so the console alert display
        # (built for probabilities) still reads sensibly for a RUL value -
        # the actual number is in `reason`, this is just for the icon/urgency line
        risk_score = max(0.0, min(1.0, 1 - (rul / (self._warning * 2))))

        return {
            "udi": row.get("index", "?"),
            "action": action,
            "urgency": urgency,
            "reason": reason,
            "failure_probability": risk_score,
        }