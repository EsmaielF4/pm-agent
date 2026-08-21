"""
P3 demo: turbine Remaining Useful Life (RUL) regression.

Evaluated by R^2 (coefficient of determination), matching MAPNA's own
P3 scoring metric - how much of the variance in actual RUL the model
explains, from 1.0 (perfect) down through 0.0 (no better than always
guessing the average) and potentially negative (worse than the average).

Run from the pm_agent/ directory:
    python demo/run_demo_p3.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

from core.agent import PredictiveMaintenanceAgent


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "config", "agent_config_p3.yaml")
    agent = PredictiveMaintenanceAgent(config_path)

    print("=" * 60)
    print("ACTIVE PLUGIN CONFIGURATION")
    print("=" * 60)
    for k, v in agent.status()["active_plugins"].items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    raw, X, y = agent.train()
    print(f"Trained on {len(raw)} rows.")
    print(f"RUL_seconds range: {y.min():.0f} to {y.max():.0f}, mean {y.mean():.0f}")

    # 5-fold cross-validated R^2 - the trustworthy number
    cv_reg = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    r2_scores = cross_val_score(cv_reg, X, y, cv=cv, scoring="r2")
    print(
        f"\n5-fold cross-validated R^2: {r2_scores.mean():.3f} "
        f"(+/- {r2_scores.std():.3f}) — folds: {[round(s, 3) for s in r2_scores]}"
    )

    # Holdout split for MAE (an interpretable "off by how many seconds" number)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1)
    agent.model.fit(Xtr, ytr)
    preds = agent.model.predict(Xte)
    yhat = [p["predicted_rul_seconds"] for p in preds]
    mae = mean_absolute_error(yte, yhat)
    holdout_r2 = r2_score(yte, yhat)
    print(f"\nHoldout R^2: {holdout_r2:.3f}")
    print(f"Holdout MAE: {mae:.1f} seconds (average prediction error)")

    print("\nTop feature importances:")
    for feat, imp in sorted(
        agent.model.feature_importances().items(), key=lambda kv: -kv[1]
    )[:8]:
        print(f"  {feat}: {imp:.3f}")

    # retrain on full data for the live demo pass
    agent.model.fit(X, y)

    print("\n" + "=" * 60)
    print(f"LIVE AGENT CYCLE — {agent.cfg['demo']['n_batch_rows']} readings")
    print("=" * 60)
    agent.run_cycle()

    print("\nDone. Three problem types now share one architecture:")
    print("  P2 = binary classification, P4 = multi-label classification,")
    print("  P3 = regression - only the plugins changed, not the core loop.")


if __name__ == "__main__":
    main()