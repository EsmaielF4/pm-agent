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
from models import fault_diagnosis  # noqa: F401
from models import rul_regressor  # noqa: F401
from models import boosting_classifiers  # noqa: F401
from actions import alerting  # noqa: F401
from core import rul_decision  # noqa: F401
from core import adaptive_decision  # noqa: F401
from core import fault_decision  # noqa: F401
from core import rul_decision  # noqa: F401


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
        # target_column may be a single column name (e.g. P2's "faulted")
        # or a list of column names (e.g. P4's ["fault_type", "fault_source"]) -
        # both are supported without any other change to this method.
        target_cols = target_col if isinstance(target_col, list) else [target_col]
        raw = raw.dropna(subset=target_cols)
        X = self.feature_extractor.transform(raw)
        y = raw[target_cols[0]] if len(target_cols) == 1 else raw[target_cols]
        self.model.fit(X, y)
        # Let the decision policy learn data-derived thresholds from the
        # model's own training-set predictions, if it supports that
        # (no-op for policies that don't - see DecisionPolicy.calibrate).
        self.decision_policy.calibrate(self.model.predict(X), y)
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
    def predict_batch(self, raw: pd.DataFrame):
        """Run the trained model on a batch of raw rows and return the
        raw predictions - no decision/alerting logic, just model output.
        Used for generating a submission-style predictions file, where
        you want a plain answer per row rather than live agent alerts."""
        X = self.feature_extractor.transform(raw)
        return self.model.predict(X)
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
