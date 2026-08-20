"""
Generate a MAPNA-style predictions file for problem P2.

Trains on the labeled Train data (same as the demo), then runs the
trained model on the real Test files - which have NO label column,
since MAPNA keeps the true answers to score submissions themselves.

Run from the pm_agent/ directory:
    python demo/generate_predictions.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import PredictiveMaintenanceAgent
from data.multisensor import MAPNAMultiSensorSource


def main():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "agent_config.yaml",
    )
    agent = PredictiveMaintenanceAgent(config_path)

    print("Training on labeled data...")
    agent.train()

    test_dir = agent.cfg["data_source"]["params"]["data_dir"].replace(
        "mapna_p2_train", "mapna_p2_test"
    )
    test_source = MAPNAMultiSensorSource(
        data_dir=test_dir,
        file_suffix="_test.csv",
        reference_sensor="VibAccel_m_s2",
    )

    print(f"Loading real test data from: {test_dir}")
    test_raw = test_source.full_data()
    print(f"Loaded {len(test_raw)} test rows (no labels - these are what we predict).")

    predictions = agent.predict_batch(test_raw)

    out = test_raw[["timestamp"]].copy() if "timestamp" in test_raw.columns else test_raw.copy()
    out["predicted_label"] = [p["predicted_label"] for p in predictions]
    out["failure_probability"] = [p["failure_probability"] for p in predictions]

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "predictions_p2.csv",
    )
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out)} predictions to: {out_path}")
    print(out.head())


if __name__ == "__main__":
    main()