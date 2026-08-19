"""
Action layer interface - the "Act" stage.

An ActionHandler receives a decision (from the agent's decide step) and
does something with it. Today: print/log. Later: send an email, open a
ticket, push to a dashboard, call a maintenance-scheduling API - all
without the agent's core loop knowing or caring which.
"""

from abc import ABC, abstractmethod


class ActionHandler(ABC):
    @abstractmethod
    def handle(self, decision: dict) -> None:
        """decision is a dict produced by the agent's decide step, e.g.
        {"udi": 5, "action": "SCHEDULE_MAINTENANCE", "urgency": "high",
         "reason": "...", "failure_probability": 0.91}"""
        raise NotImplementedError
