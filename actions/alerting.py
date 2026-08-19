"""
Concrete ActionHandler plugins.
"""

from core.registry import register
from actions.base import ActionHandler


@register("action", "console_alert")
class ConsoleAlertHandler(ActionHandler):
    """Prints a formatted alert. Stand-in for week 1; swap for an
    EmailAlertHandler / DashboardPushHandler / TicketHandler later by
    adding a class + one config line."""

    _ICONS = {"low": "🟢", "medium": "🟡", "high": "🔴"}

    def handle(self, decision: dict) -> None:
        icon = self._ICONS.get(decision.get("urgency", "low"), "⚪")
        print(
            f"{icon} [UDI {decision['udi']}] {decision['action']} "
            f"(urgency={decision['urgency']}, "
            f"p_failure={decision['failure_probability']:.2f}) "
            f"- {decision['reason']}"
        )


@register("action", "log_file")
class LogFileHandler(ActionHandler):
    """Appends decisions to a log file - useful once the demo runs
    unattended or you want an audit trail for the presentation."""

    def __init__(self, path: str = "agent_decisions.log"):
        self._path = path

    def handle(self, decision: dict) -> None:
        with open(self._path, "a") as f:
            f.write(str(decision) + "\n")
