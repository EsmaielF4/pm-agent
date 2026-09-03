"""
Gradient Boosting classifiers - XGBoost and LightGBM - for P2 pump
fault detection. Both implement the same Predictor interface as
RandomForestFailureClassifier, so they're drop-in comparable/
swappable via config, and both slot into the existing DecisionPolicy
and dashboard code unchanged.

Why these exist alongside Random Forest: Random Forest builds many
trees independently and averages their votes. Gradient Boosting builds
trees one at a time, where each new tree specifically targets the
current ensemble's mistakes - a more targeted error-correction process
that often (not always) edges out Random Forest on tabular data. See
demo/benchmark_models.py for a head-to-head comparison on real data.

Both classes handle string class labels (e.g. "faulted"/"normal")
manually via their own label encoding, rather than relying on each
library's built-in string-label handling - that behavior has changed
across versions of both libraries, so encoding it ourselves keeps this
plugin's behavior stable regardless of which version gets installed.
"""

import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from core.registry import register
from models.base import Predictor


class _BoostingClassifierBase(Predictor):
    """Shared label-encoding logic for both boosting plugins."""

    def __init__(self, positive_label: str = None):
        self._positive_label = positive_label
        self._label_to_int = None
        self._int_to_label = None
        self._positive_int = 1
        self._fitted = False
        self._feature_names = None
        self._clf = None  # set by subclass

    def fit(self, X: pd.DataFrame, y) -> None:
        self._feature_names = list(X.columns)
        y = pd.Series(y).reset_index(drop=True)
        classes = sorted(y.unique())

        if self._positive_label and self._positive_label in classes and len(classes) == 2:
            other = [c for c in classes if c != self._positive_label][0]
            self._label_to_int = {other: 0, self._positive_label: 1}
        else:
            self._label_to_int = {c: i for i, c in enumerate(classes)}
        self._int_to_label = {v: k for k, v in self._label_to_int.items()}
        self._positive_int = self._label_to_int.get(self._positive_label, 1)

        y_encoded = y.map(self._label_to_int)
        self._clf.fit(X, y_encoded)
        self._fitted = True

    def predict(self, X: pd.DataFrame) -> list:
        if not self._fitted:
            raise RuntimeError("Model not fitted yet - call fit() first.")
        probs = self._clf.predict_proba(X)
        preds_int = self._clf.predict(X)
        return [
            {
                "failure_probability": float(p[self._positive_int]),
                "predicted_label": self._int_to_label[int(lbl)],
            }
            for p, lbl in zip(probs, preds_int)
        ]

    def feature_importances(self) -> dict:
        if not self._fitted:
            return {}
        return dict(zip(self._feature_names, self._clf.feature_importances_))


@register("model", "xgb_failure_classifier")
class XGBoostFailureClassifier(_BoostingClassifierBase):
    def __init__(self, n_estimators: int = 300, random_state: int = 42, positive_label: str = "faulted"):
        super().__init__(positive_label=positive_label)
        self._clf = XGBClassifier(
            n_estimators=n_estimators, random_state=random_state, eval_metric="logloss"
        )

    def name(self) -> str:
        return "xgb_failure_classifier"


@register("model", "lgbm_failure_classifier")
class LightGBMFailureClassifier(_BoostingClassifierBase):
    def __init__(self, n_estimators: int = 300, random_state: int = 42, positive_label: str = "faulted"):
        super().__init__(positive_label=positive_label)
        self._clf = LGBMClassifier(
            n_estimators=n_estimators, random_state=random_state,
            class_weight="balanced", verbose=-1,
        )

    def name(self) -> str:
        return "lgbm_failure_classifier"