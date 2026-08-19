"""
The Agent: assembles whichever plugins are named in config, and runs
the Perceive -> Predict -> Decide -> Act loop. This file should almost
never need to change as you add capabilities - it just reads config
and asks the registry for instances.
"""

import yaml
import pandas as pd

from core import registry
# Import plugin modules so their @register decorators run and populate
# the registry. Adding a new plugin file here is the only "wiring" step
# needed when you create a new one.
from data import sources  # noqa: F401
from data import multisensor  # noqa: F401
from features import engineering  # noqa: F401
from features import mapna_features  # noqa: F401
from models import failure_classifier  # noqa: F401
from actions import alerting  # noqa: F401
from core import decision  # noqa: F401


class PredictiveMaintenanceAgent:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.data_source = registry.get("data_source", self.cfg["data_source"]["name"])(
            **self.cfg["data_source"]["params"]
        )
        self.feature_extractor = registry.get(
            "feature_extractor", self.cfg["feature_extractor"]["name"]
        )(**self.cfg["feature_extractor"]["params"])
        self.model = registry.get("model", self.cfg["model"]["name"])(
            **self.cfg["model"]["params"]
        )
        self.decision_policy = registry.get(
            "decision_policy", self.cfg["decision_policy"]["name"]
        )(**self.cfg["decision_policy"]["params"])
        self.action_handlers = [
            registry.get("action", h["name"])(**h["params"])
            for h in self.cfg["action_handlers"]
        ]

    def train(self):
        target_col = self.cfg["training"]["target_column"]
        if hasattr(self.data_source, "full_data"):
            raw = self.data_source.full_data()
        else:
            raw = self.data_source.get_batch(self.cfg["training"]["n_training_rows"])
        raw = raw.dropna(subset=[target_col])
        X = self.feature_extractor.transform(raw)
        y = raw[target_col]
        self.model.fit(X, y)
        return raw, X, y

    def run_cycle(self, n: int = None):
        """One Perceive -> Predict -> Decide -> Act pass over a batch."""
        n = n or self.cfg["demo"]["n_batch_rows"]
        raw = self.data_source.get_batch(n)                # Perceive
        X = self.feature_extractor.transform(raw)           # Feature layer
        predictions = self.model.predict(X)                 # Predict

        decisions = []
        for (_, row), pred in zip(raw.iterrows(), predictions):
            decision_ = self.decision_policy.decide(row.to_dict(), pred)  # Decide
            for handler in self.action_handlers:
                handler.handle(decision_)                    # Act
            decisions.append(decision_)

        return raw, predictions, decisions

    def status(self):
        return {
            "active_plugins": {
                "data_source": self.cfg["data_source"]["name"],
                "feature_extractor": self.cfg["feature_extractor"]["name"],
                "model": self.cfg["model"]["name"],
                "decision_policy": self.cfg["decision_policy"]["name"],
                "action_handlers": [h["name"] for h in self.cfg["action_handlers"]],
            },
            "all_registered_plugins": registry.all_registered(),
        }
