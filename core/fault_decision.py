"""
Decision policy for fault diagnosis (P4).

Different reasoning shape than P2's threshold policy: instead of "how
likely is failure", this equipment is ALREADY known to be faulty (P4's
whole dataset is faulty rows) - the question is "who do we send, and
how urgently". This is the concrete point of separating the Decide
stage from the Predictor: a structurally different task still slots
into the same Agent loop, just with a different DecisionPolicy plugin.
"""

from core.registry import register
from core.decision import DecisionPolicy


@register("decision_policy", "fault_routing_policy")
class FaultRoutingPolicy(DecisionPolicy):
    """Routes based on fault_source first (equipment vs. sensor problem),
    then fault_type for equipment faults. Low-confidence diagnoses are
    flagged for human review rather than acted on automatically -
    exactly the kind of domain judgment this layer exists for."""

    _TYPE_TEAM = {
        "bearing_fault": "mechanical team - bearing inspection",
        "misalignment": "mechanical team - alignment check",
        "lubrication_fault": "maintenance team - lubrication system",
        "sensor_fault": "instrumentation team",
    }

    def __init__(self, min_confidence: float = 0.5):
        self._min_confidence = min_confidence

    def decide(self, row: dict, prediction: dict) -> dict:
        fault_type = prediction["predicted_fault_type"]
        fault_source = prediction["predicted_fault_source"]
        type_conf = prediction["fault_type_confidence"]
        source_conf = prediction["fault_source_confidence"]

        if source_conf < self._min_confidence or type_conf < self._min_confidence:
            action = "FLAG_FOR_HUMAN_REVIEW"
            urgency = "medium"
            reason = (
                f"Low confidence diagnosis (type={type_conf:.0%}, "
                f"source={source_conf:.0%}) - needs a person to confirm"
            )
        elif fault_source == "sensor_fault":
            action = "DISPATCH_INSTRUMENTATION_TEAM"
            urgency = "low"
            reason = "Diagnosed as a sensor fault, not an equipment fault - check instrumentation"
        else:
            team = self._TYPE_TEAM.get(fault_type, "maintenance team")
            action = f"DISPATCH_{fault_type.upper()}"
            urgency = "high"
            reason = f"Equipment fault diagnosed as {fault_type} ({type_conf:.0%} confidence) - route to {team}"

        return {
            "udi": row.get("timestamp"),
            "action": action,
            "urgency": urgency,
            "reason": reason,
            "failure_probability": type_conf,  # reused key so ActionHandler works unchanged
        }