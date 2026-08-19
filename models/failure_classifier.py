"""
Baseline failure-probability classifier.

Random Forest chosen deliberately for week 1: fast to train, robust on
small/synthetic tabular data, gives feature importances for free (useful
for the explainability plugin later), and needs no GPU/tuning to get a
working demo. This is meant to be replaced or joined by better models
(gradient boosting, a proper time-series model once you have sequential
field data) - it's just one plugin in the registry.
"""

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from core.registry import register
from models.base import Predictor


@register("model", "rf_failure_classifier")
class RandomForestFailureClassifier(Predictor):
    def __init__(self, n_estimators: int = 200, random_state: int = 42, positive_label=1):
        """positive_label: the class value that means "failure" - e.g. 1
        for the synthetic AI4I data, or "faulted" for the MAPNA dataset.
        Needed because predict_proba's column order follows sorted class
        labels, not which one is semantically "bad"."""
        self._clf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, class_weight="balanced"
        )
        self._fitted = False
        self._feature_names = None
        self._positive_label = positive_label
        self._pos_idx = None

    def fit(self, X: pd.DataFrame, y) -> None:
        self._feature_names = list(X.columns)
        self._clf.fit(X, y)
        self._pos_idx = list(self._clf.classes_).index(self._positive_label)
        self._fitted = True

    def predict(self, X: pd.DataFrame) -> list:
        if not self._fitted:
            raise RuntimeError("Model not fitted yet - call fit() first.")
        probs = self._clf.predict_proba(X)[:, self._pos_idx]
        preds = self._clf.predict(X)
        return [
            {"failure_probability": float(p), "predicted_label": lbl}
            for p, lbl in zip(probs, preds)
        ]

    def feature_importances(self) -> dict:
        if not self._fitted:
            return {}
        return dict(zip(self._feature_names, self._clf.feature_importances_))

    def name(self) -> str:
        return "rf_failure_classifier"
