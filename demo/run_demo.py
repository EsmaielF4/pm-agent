"""
End-to-end demo: build the agent from config, train it, then run it on
a fresh batch of "incoming" readings and watch it make decisions.

Run from the pm_agent/ directory:
    python demo/run_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import classification_report

from core.agent import PredictiveMaintenanceAgent


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_arg = sys.argv[1] if len(sys.argv) > 1 else "agent_config.yaml"
    config_name = os.path.basename(config_arg)
    config_path = os.path.join(project_root, "config", config_name)
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
    print("Label distribution:")
    print(y.value_counts(normalize=True).mul(100).round(1).astype(str) + "%")

    # Cross-validated evaluation: more trustworthy than a single random
    # split, since one split's score can shift depending on which rows
    # happen to land in the test slice. 5-fold CV trains/tests 5 times
    # on different slices and reports the spread, not just one number.
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.ensemble import RandomForestClassifier

    cv_clf = RandomForestClassifier(
        n_estimators=agent.cfg["model"]["params"].get("n_estimators", 300),
        random_state=42,
        class_weight="balanced",
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    cv_scores = cross_val_score(cv_clf, X, y, cv=cv, scoring="f1_macro")
    print(
        f"\n5-fold cross-validated Macro-F1: {cv_scores.mean():.3f} "
        f"(+/- {cv_scores.std():.3f}) — folds: {[round(s, 3) for s in cv_scores]}"
    )

    # A single holdout split too, for a detailed per-class breakdown
    # (precision/recall) - the CV score above is the number to trust,
    # this is just to see WHERE mistakes happen.
    from sklearn.model_selection import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)
    agent.model.fit(Xtr, ytr)
    preds = agent.model.predict(Xte)
    yhat = [p["predicted_label"] for p in preds]
    print("\nPer-class breakdown (single holdout split, for detail only):")
    print(classification_report(yte, yhat, digits=3, zero_division=0))

    print("Top feature importances:")
    for feat, imp in sorted(
        agent.model.feature_importances().items(), key=lambda kv: -kv[1]
    )[:8]:
        print(f"  {feat}: {imp:.3f}")

    # retrain on full data for the live demo pass
    agent.model.fit(X, y)

    print("\n" + "=" * 60)
    print(f"LIVE AGENT CYCLE — {agent.cfg['demo']['n_batch_rows']} new readings")
    print("=" * 60)
    agent.run_cycle()

    print("\nDone. Edit config/agent_config.yaml to swap any plugin;")
    print("add a new plugin class + @register(...) to extend a capability.")


if __name__ == "__main__":
    main()
