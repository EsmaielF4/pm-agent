"""
Remaining Useful Life (RUL) regressor - MAPNA P3.

Structurally different again from P2 (binary) and P4 (multi-label
classification): this predicts a CONTINUOUS number (seconds until
failure), not a category. Still fits the same Predictor interface
(fit/predict/name) - the agent's core loop doesn't care whether a
plugin outputs a class or a number.

Note on features: P3's rows are NOT one continuous time series - the
RUL_seconds values jump around non-monotonically row to row, which
means this data is many separate equipment life-cycles concatenated
together with no ID column marking the boundaries. Rolling/trend
features (safe for P2, which IS one continuous stream) would be unsafe
here - a rolling window could blend two unrelated cycles at a boundary
we have no way to detect. So P3 deliberately uses only raw
instantaneous sensor readings as features (see config/agent_config_p3.yaml).
"""

from sklearn.ensemble import RandomForestRegressor
import pandas as pd

from core.registry import register
from models.base import Predictor


@register("model", "rul_regressor")
class RULRegressor(Predictor):
    def __init__(self, n_estimators: int = 300, random_state: int = 42):
        self._reg = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        )
        self._fitted = False
        self._feature_names = None

    def fit(self, X: pd.DataFrame, y) -> None:
        self._feature_names = list(X.columns)
        self._reg.fit(X, y)
        self._fitted = True

    def predict(self, X: pd.DataFrame) -> list:
        if not self._fitted:
            raise RuntimeError("Model not fitted yet - call fit() first.")
        preds = self._reg.predict(X)
        return [{"predicted_rul_seconds": float(p)} for p in preds]

    def feature_importances(self) -> dict:
        if not self._fitted:
            return {}
        return dict(zip(self._feature_names, self._reg.feature_importances_))

    def name(self) -> str:
        return "rul_regressor"