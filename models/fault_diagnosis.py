"""
Multi-label fault diagnosis predictor (MAPNA P4).

Structurally different from the P2 binary classifier: instead of one
yes/no failure flag, this predicts TWO separate labels per row -
fault_type (which component/mode: bearing, misalignment, lubrication,
sensor) and fault_source (equipment_fault vs sensor_fault, i.e. "is the
equipment actually broken, or is a SENSOR just reporting garbage").

That second distinction matters a lot in practice: a sensor_fault means
send someone to check the instrumentation, not tear down the pump.

Implementation: two independent RandomForestClassifiers sharing the
same feature matrix - simplest thing that works, and still fits the
existing Predictor interface (fit/predict/name), which is exactly the
point: the agent's core loop doesn't need to know or care that this
plugin predicts two things instead of one.
"""

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from core.registry import register
from models.base import Predictor


@register("model", "fault_diagnosis_classifier")
class FaultDiagnosisClassifier(Predictor):
    def __init__(self, n_estimators: int = 300, random_state: int = 42):
        self._type_clf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, class_weight="balanced"
        )
        self._source_clf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, class_weight="balanced"
        )
        self._fitted = False
        self._feature_names = None

    def fit(self, X: pd.DataFrame, y) -> None:
        """y must be a DataFrame with columns ['fault_type', 'fault_source']
        (not a single column - this predictor needs both labels)."""
        self._feature_names = list(X.columns)
        self._type_clf.fit(X, y["fault_type"])
        self._source_clf.fit(X, y["fault_source"])
        self._fitted = True

    def predict(self, X: pd.DataFrame) -> list:
        if not self._fitted:
            raise RuntimeError("Model not fitted yet - call fit() first.")
        type_preds = self._type_clf.predict(X)
        type_probs = self._type_clf.predict_proba(X)
        source_preds = self._source_clf.predict(X)
        source_probs = self._source_clf.predict_proba(X)

        type_classes = self._type_clf.classes_
        source_classes = self._source_clf.classes_

        out = []
        for i in range(len(X)):
            out.append(
                {
                    "predicted_fault_type": type_preds[i],
                    "fault_type_confidence": float(type_probs[i].max()),
                    "fault_type_probs": dict(zip(type_classes, type_probs[i].tolist())),
                    "predicted_fault_source": source_preds[i],
                    "fault_source_confidence": float(source_probs[i].max()),
                }
            )
        return out

    def feature_importances(self) -> dict:
        if not self._fitted:
            return {}
        return {
            "fault_type": dict(zip(self._feature_names, self._type_clf.feature_importances_)),
            "fault_source": dict(
                zip(self._feature_names, self._source_clf.feature_importances_)
            ),
        }

    def name(self) -> str:
        return "fault_diagnosis_classifier"