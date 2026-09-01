"""
Predictive Maintenance Agent - interactive dashboard.

Wraps the SAME Agent/plugin architecture used by the CLI demos - this
file contains no modeling logic of its own, only visualization. That's
deliberate: it's proof the architecture supports a completely different
"Act" surface (a web UI instead of console prints) without touching
core/agent.py, core/registry.py, or any model/data plugin.

Run from the pm_agent/ directory:
    streamlit run dashboard/app.py
"""

import os
import sys
import tempfile

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import PredictiveMaintenanceAgent

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROBLEMS = {
    "P2 — Pump Fault Detection": {
        "config": "config/agent_config.yaml",
        "type": "binary",
        "sensors": ["Temperature_C", "Pressure_kPa", "VibAccel_m_s2", "VibVelocity_mm_s"],
        "label_hint": '"faulted" (values like "faulted"/"normal")',
    },
    "P3 — Turbine Remaining Useful Life": {
        "config": "config/agent_config_p3.yaml",
        "type": "regression",
        "sensors": [
            "BearingTemp_C", "Current_A", "FlowRate_L_min", "Humidity_pct",
            "OilLevel_cm", "Power_kW", "Pressure_kPa", "Speed_RPM",
            "Temperature_C", "Torque_Nm", "VibAccel_m_s2", "VibDisp_mm",
            "VibVelocity_mm_s", "Voltage_V", "pH_units",
        ],
        "label_hint": '"RUL_seconds" (a number)',
    },
    "P4 — Fault Type & Source Diagnosis": {
        "config": "config/agent_config_p4.yaml",
        "type": "multilabel",
        "sensors": [
            "BearingTemp_C", "FlowRate_L_min", "Humidity_pct", "OilLevel_cm",
            "Power_kW", "Pressure_kPa", "Speed_RPM", "Temperature_C",
            "Torque_Nm", "VibAccel_m_s2", "VibDisp_mm", "VibVelocity_mm_s",
            "Voltage_V", "pH_units",
        ],
        "label_hint": '"fault_type" and "fault_source"',
    },
}

st.set_page_config(page_title="Predictive Maintenance Agent", layout="wide")
st.title("🔧 Predictive Maintenance Agent")
st.caption(
    "Same Perceive → Predict → Decide → Act architecture as the CLI demos, "
    "just visualized. Pick a problem, train it, and watch it monitor live readings."
)

with st.sidebar:
    st.header("1. Choose a problem")
    problem_name = st.selectbox("Problem", list(PROBLEMS.keys()))
    problem = PROBLEMS[problem_name]

    st.header("2. Choose data")
    data_mode = st.radio("Data source", ["Use bundled MAPNA data", "Upload my own CSVs"])

    uploaded_files = None
    if data_mode == "Upload my own CSVs":
        sensor_list = "\n".join(f"- `{s}_train.csv`" for s in problem["sensors"])
        st.markdown(
            f"Upload one CSV per sensor, named exactly as below. "
            f"**Exactly one** of these files must also include the label "
            f"column(s): {problem['label_hint']}.\n\n{sensor_list}"
        )
        uploaded_files = st.file_uploader(
            "Sensor CSV files", type="csv", accept_multiple_files=True
        )

    run_clicked = st.button("🚀 Train & Run Agent", type="primary")

# Reset stored results if the problem changed since the last run
if st.session_state.get("last_problem") != problem_name:
    st.session_state.pop("agent", None)

if run_clicked:
    config_path = os.path.join(PROJECT_ROOT, problem["config"])
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if data_mode == "Upload my own CSVs":
        if not uploaded_files:
            st.error("Please upload sensor CSV files first.")
            st.stop()
        tmp_dir = tempfile.mkdtemp(prefix="pm_agent_upload_")
        for uf in uploaded_files:
            with open(os.path.join(tmp_dir, uf.name), "wb") as out:
                out.write(uf.getbuffer())
        cfg["data_source"]["params"]["data_dir"] = tmp_dir
        config_path = os.path.join(tmp_dir, "_dashboard_config.yaml")
        with open(config_path, "w") as out:
            yaml.safe_dump(cfg, out)

    try:
        with st.spinner("Training agent..."):
            agent = PredictiveMaintenanceAgent(config_path)
            raw, X, y = agent.train()
        st.session_state["agent"] = agent
        st.session_state["raw"] = raw
        st.session_state["X"] = X
        st.session_state["y"] = y
        st.session_state["problem_type"] = problem["type"]
        st.session_state["last_problem"] = problem_name
    except Exception as e:
        st.error(f"Something went wrong loading or training on this data: {e}")
        st.stop()

if "agent" in st.session_state:
    agent = st.session_state["agent"]
    raw, X, y = st.session_state["raw"], st.session_state["X"], st.session_state["y"]
    ptype = st.session_state["problem_type"]

    st.success(f"Trained on {len(raw):,} rows.")

    # ---- Metrics, per problem type ----
    st.subheader("📊 Model performance")
    col1, col2, col3 = st.columns(3)

    if ptype == "binary":
        from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
        scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")
        col1.metric("5-fold Macro-F1", f"{scores.mean():.3f}", f"± {scores.std():.3f}")

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)
        agent.model.fit(Xtr, ytr)
        preds = agent.model.predict(Xte)
        yhat = [p["predicted_label"] for p in preds]
        report = classification_report(yte, yhat, output_dict=True, zero_division=0)
        col2.metric("Holdout accuracy", f"{report['accuracy']:.3f}")
        fault_labels = [c for c in report if c not in ("accuracy", "macro avg", "weighted avg")]
        if fault_labels:
            col3.metric(f"'{fault_labels[0]}' recall", f"{report[fault_labels[0]]['recall']:.3f}")

    elif ptype == "regression":
        from sklearn.model_selection import KFold, cross_val_score, train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error

        cv = KFold(n_splits=5, shuffle=True, random_state=1)
        reg = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        scores = cross_val_score(reg, X, y, cv=cv, scoring="r2")
        col1.metric("5-fold R²", f"{scores.mean():.3f}", f"± {scores.std():.3f}")

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1)
        agent.model.fit(Xtr, ytr)
        preds = agent.model.predict(Xte)
        yhat = [p["predicted_rul_seconds"] for p in preds]
        col2.metric("MAE", f"{mean_absolute_error(yte, yhat):.1f}s")
        col3.metric("Rows trained", f"{len(raw):,}")

    elif ptype == "multilabel":
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score
        from models.fault_diagnosis import FaultDiagnosisClassifier

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=1, stratify=y["fault_type"]
        )
        eval_model = FaultDiagnosisClassifier(n_estimators=300)
        eval_model.fit(Xtr, ytr)
        preds = eval_model.predict(Xte)
        type_f1 = f1_score(yte["fault_type"], [p["predicted_fault_type"] for p in preds], average="macro")
        source_f1 = f1_score(yte["fault_source"], [p["predicted_fault_source"] for p in preds], average="macro")
        col1.metric("fault_type Macro-F1", f"{type_f1:.3f}")
        col2.metric("fault_source Macro-F1", f"{source_f1:.3f}")
        col3.metric("Combined score", f"{(type_f1 + source_f1) / 2:.3f}")

    # retrain on the full dataset for the live section (also calibrates
    # the decision policy, if it supports that - see core/agent.py train())
    agent.model.fit(X, y)
    agent.decision_policy.calibrate(agent.model.predict(X), y)

    # ---- Feature importance ----
    st.subheader("🔍 What the model is looking at")
    importances = agent.model.feature_importances()
    if ptype == "multilabel":
        importances = importances.get("fault_type", {})
    imp_items = sorted(importances.items(), key=lambda kv: -kv[1])[:10]
    imp_df = pd.DataFrame(imp_items, columns=["feature", "importance"]).set_index("feature")
    st.bar_chart(imp_df)

    # ---- Live monitoring / alerts feed ----
    st.subheader("🚨 Live monitoring — alerts & notifications")
    n_live = st.slider("Number of readings to simulate", 5, 50, 20)
    raw_batch, predictions, decisions = agent.run_cycle(n=n_live)

    urgency_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}
    counts = {"high": 0, "medium": 0, "low": 0}
    for d in decisions:
        counts[d["urgency"]] = counts.get(d["urgency"], 0) + 1

    m1, m2, m3 = st.columns(3)
    m1.metric("🔴 High urgency", counts["high"])
    m2.metric("🟡 Medium urgency", counts["medium"])
    m3.metric("🟢 Monitoring normally", counts["low"])

    for d in reversed(decisions):
        icon = urgency_icon.get(d["urgency"], "⚪")
        with st.container(border=True):
            st.markdown(f"{icon} **{d['action']}**  ·  `{d['udi']}`")
            st.caption(d["reason"])

    # ---- Primary sensor trend, with alert markers ----
    st.subheader("📈 Primary sensor trend")
    if imp_items:
        primary = imp_items[0][0].replace("_roll_mean", "").replace("_roll_std", "")
        if primary in raw_batch.columns:
            fig, ax = plt.subplots(figsize=(10, 3))
            y_vals = raw_batch[primary].reset_index(drop=True)
            ax.plot(range(len(y_vals)), y_vals, marker="o", linewidth=1, color="#4C72B0")
            for i, d in enumerate(decisions):
                if d["urgency"] == "high":
                    ax.scatter(i, y_vals.iloc[i], color="red", s=90, zorder=5)
                elif d["urgency"] == "medium":
                    ax.scatter(i, y_vals.iloc[i], color="orange", s=60, zorder=5)
            ax.set_ylabel(primary)
            ax.set_xlabel("Reading #")
            st.pyplot(fig)
else:
    st.info("Choose a problem and data source in the sidebar, then click **Train & Run Agent**.")