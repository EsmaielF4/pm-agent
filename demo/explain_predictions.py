"""
SHAP-based explainability: for a handful of live readings, show WHICH
features drove each SPECIFIC prediction - not just an aggregate
importance ranking (feature_importances_ already gives you that), but
a per-prediction breakdown. This is the concrete implementation of the
"explainability reduces alarm fatigue" finding from the maintenance
domain research (see docs/ for the write-up).

Run from the pm_agent/ directory:
    python demo/explain_predictions.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shap

from core.agent import PredictiveMaintenanceAgent


def _extract_positive_class_values(shap_values, positive_index=1):
    """SHAP's output shape has varied across versions/model types -
    normalize it to a single (rows x features) array for the positive
    class either way."""
    if isinstance(shap_values, list):
        return shap_values[positive_index]
    if hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        return shap_values[:, :, positive_index]
    return shap_values


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "config", "agent_config.yaml")
    agent = PredictiveMaintenanceAgent(config_path)

    print("Training model...")
    raw, X, y = agent.train()

    print("Building SHAP explainer for the trained model...")
    underlying_model = agent.model._clf
    explainer = shap.TreeExplainer(underlying_model)

    print("Getting a few live readings to explain...\n")
    raw_batch = agent.data_source.get_batch(5)
    X_batch = agent.feature_extractor.transform(raw_batch)
    predictions = agent.model.predict(X_batch)

    shap_values_raw = explainer.shap_values(X_batch)
    shap_values = _extract_positive_class_values(shap_values_raw)
    feature_names = list(X_batch.columns)

    for i in range(len(X_batch)):
        pred = predictions[i]
        print("=" * 60)
        print(
            f"Reading {i + 1}: predicted '{pred['predicted_label']}' "
            f"(failure_probability={pred['failure_probability']:.2f})"
        )
        print("Why - top contributing features for THIS reading:")
        contribs = sorted(
            zip(feature_names, shap_values[i]), key=lambda c: -abs(c[1])
        )
        for feat, val in contribs[:5]:
            direction = "toward faulted" if val > 0 else "toward normal"
            print(f"  {feat:30s} {val:+.4f}  ({direction})")
        print()


if __name__ == "__main__":
    main()