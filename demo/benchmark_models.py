"""
Benchmark: Random Forest vs XGBoost vs LightGBM on P2 pump fault
detection, using the SAME rolling-trend features and the SAME 5-fold
cross-validation methodology as demo/run_demo.py - so this is a fair,
apples-to-apples comparison against the 0.968 Macro-F1 baseline.

Run from the pm_agent/ directory:
    python demo/benchmark_models.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from core.agent import PredictiveMaintenanceAgent


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "config", "agent_config.yaml")
    agent = PredictiveMaintenanceAgent(config_path)

    print("Loading and preparing data (same pipeline as demo/run_demo.py)...")
    raw = agent.data_source.full_data()
    target_col = agent.cfg["training"]["target_column"]
    raw = raw.dropna(subset=[target_col])
    X = agent.feature_extractor.transform(raw)
    y_raw = raw[target_col]

    classes = sorted(y_raw.unique())
    positive_label = "faulted" if "faulted" in classes else classes[-1]
    other = [c for c in classes if c != positive_label][0]
    y = y_raw.map({other: 0, positive_label: 1})

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    candidates = {
        "Random Forest (current baseline)": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(n_estimators=300, random_state=42, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(
            n_estimators=300, random_state=42, class_weight="balanced", verbose=-1
        ),
    }

    print(f"\nRunning 5-fold cross-validation on {len(X)} rows, {X.shape[1]} features...\n")
    results = []
    for name, clf in candidates.items():
        t0 = time.time()
        scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")
        elapsed = time.time() - t0
        results.append((name, scores.mean(), scores.std(), elapsed))
        print(f"{name:35s}  Macro-F1 = {scores.mean():.4f}  (+/- {scores.std():.4f})   [{elapsed:.1f}s]")

    print("\n" + "=" * 70)
    best = max(results, key=lambda r: r[1])
    baseline = results[0][1]
    print(f"Best: {best[0]}  ->  Macro-F1 = {best[1]:.4f}")
    if best[0] != results[0][0]:
        print(f"That's a {best[1] - baseline:+.4f} improvement over the current Random Forest baseline.")
    else:
        print("The current Random Forest baseline remains the best performer here.")
    print("=" * 70)


if __name__ == "__main__":
    main()