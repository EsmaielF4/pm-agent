"""
P4 demo: fault type + fault source diagnosis.

Evaluated the same way MAPNA's own P4 problem is scored: the AVERAGE of
two separate Macro-F1 scores (one for fault_type, one for fault_source) -
not a single accuracy number, since these are two distinct label sets.

Run from the pm_agent/ directory:
    python demo/run_demo_p4.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, classification_report

from core.agent import PredictiveMaintenanceAgent
from models.fault_diagnosis import FaultDiagnosisClassifier


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "config", "agent_config_p4.yaml")
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
    print(f"Trained on {len(raw)} rows (all already-faulty equipment).")
    print("\nfault_type distribution:")
    print(y["fault_type"].value_counts())
    print("\nfault_source distribution:")
    print(y["fault_source"].value_counts())

    # Holdout split, stratified on fault_type (the finer-grained label)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=1, stratify=y["fault_type"]
    )
    eval_model = FaultDiagnosisClassifier(n_estimators=300)
    eval_model.fit(Xtr, ytr)
    preds = eval_model.predict(Xte)
    type_pred = [p["predicted_fault_type"] for p in preds]
    source_pred = [p["predicted_fault_source"] for p in preds]

    type_f1 = f1_score(yte["fault_type"], type_pred, average="macro")
    source_f1 = f1_score(yte["fault_source"], source_pred, average="macro")
    combined = (type_f1 + source_f1) / 2

    print("\n" + "=" * 60)
    print("EVALUATION (matches MAPNA's own P4 scoring: avg of two Macro-F1s)")
    print("=" * 60)
    print(f"fault_type  Macro-F1: {type_f1:.3f}")
    print(f"fault_source Macro-F1: {source_f1:.3f}")
    print(f"COMBINED SCORE (average): {combined:.3f}")

    print("\nfault_type per-class breakdown:")
    print(classification_report(yte["fault_type"], type_pred, digits=3, zero_division=0))
    print("fault_source per-class breakdown:")
    print(classification_report(yte["fault_source"], source_pred, digits=3, zero_division=0))

    # retrain on full data for the live demo pass
    agent.model.fit(X, y)

    print("=" * 60)
    print(f"LIVE AGENT CYCLE — {agent.cfg['demo']['n_batch_rows']} readings")
    print("=" * 60)
    agent.run_cycle()

    print("\nDone. Compare to demo/run_demo.py (P2) - same Agent loop,")
    print("different Predictor + DecisionPolicy plugins via config.")


if __name__ == "__main__":
    main()