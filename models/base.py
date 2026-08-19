"""
Model layer interface - the "Predict" stage.

A Predictor takes a feature matrix and returns predictions. Deliberately
generic (predict() returns a dict per row) so that different kinds of
models - a failure classifier, a remaining-useful-life regressor, an
anomaly detector - can all implement it and run side by side in the
agent's pipeline without special-casing.
"""

from abc import ABC, abstractmethod
import pandas as pd


class Predictor(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> list:
        """Return a list of dicts, one per row, e.g.
        {"failure_probability": 0.83, "predicted_label": 1}
        Every predictor must include enough info for the Decide stage
        to act on, but the exact keys are predictor-specific - the
        agent's decision rules just read whatever keys they need."""
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
