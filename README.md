# Predictive Maintenance Agent

A failure-prediction **agent**, not just a model. The distinction matters:
a model outputs a probability or a number; an agent perceives, predicts,
decides, and acts. This project proves that one architecture can handle
genuinely different predictive-maintenance task types — binary failure
detection, multi-label fault diagnosis, and remaining-useful-life
regression — by swapping plugins, not rewriting code.

Built on real industrial sensor data from MAPNA's rotary-equipment
predictive-maintenance competition (pumps, turbines, compressors).

## The loop

Perceive → Predict → Decide → Act
(data) (models) (agent) (output)


Every stage is an interface (`base.py` in each folder) with one or more
registered implementations (`core/registry.py`). The `Agent`
(`core/agent.py`) only ever reads a config YAML file and asks the
registry for whatever plugin is named there — it never imports a
concrete class directly. Adding a new capability means writing one new
class and adding one config line, not touching existing code.

## Project structure

pm_agent/
├── config/
│ ├── agent_config.yaml <- P2: pump fault detection (main config)
│ ├── agent_config_p3.yaml <- P3: turbine RUL regression
│ └── agent_config_p4.yaml <- P4: fault type/source diagnosis
├── core/
│ ├── registry.py <- plugin registration mechanism
│ ├── agent.py <- orchestrator, runs the loop
│ ├── decision.py <- P2's threshold-based decision policy
│ ├── fault_decision.py <- P4's fault-routing decision policy
│ └── rul_decision.py <- P3's RUL-based decision policy
├── data/
│ ├── base.py <- DataSource interface
│ ├── sources.py <- synthetic AI4I generator (early reference)
│ └── multisensor.py <- real MAPNA multi-rate sensor loader
├── features/
│ ├── base.py <- FeatureExtractor interface
│ ├── engineering.py <- AI4I feature set (early reference)
│ └── mapna_features.py <- MAPNA feature sets (raw + rolling)
├── models/
│ ├── base.py <- Predictor interface
│ ├── failure_classifier.py <- P2: binary fault classifier
│ ├── fault_diagnosis.py <- P4: multi-label fault diagnosis
│ └── rul_regressor.py <- P3: RUL regressor
├── actions/
│ ├── base.py <- ActionHandler interface
│ └── alerting.py <- console + log-file handlers (used by all 3)
└── demo/
├── run_demo.py <- runs P2
├── run_demo_p3.py <- runs P3
├── run_demo_p4.py <- runs P4
└── generate_predictions.py <- generates real submission predictions (P2)


## Run it

pip install -r requirements.txt
python demo/run_demo.py # P2: pump fault detection
python demo/run_demo_p3.py # P3: turbine RUL regression
python demo/run_demo_p4.py # P4: fault type/source diagnosis


## Results (real MAPNA data, honestly evaluated)

| Problem | Task | Metric | Score |
|---|---|---|---|
| **P2** | Pump fault detection (binary) | 5-fold CV Macro-F1 | **0.968** |
| **P3** | Turbine remaining useful life (regression) | 5-fold CV R² | **0.55** (MAE ≈ 37s) |
| **P4** | Fault type + source diagnosis (multi-label) | Avg of two Macro-F1 (MAPNA's own metric) | **0.658** |

A few honest notes on these numbers:
- P2's score comes from **rolling trend features** (moving mean/std over
  the last 5 readings per sensor) — raw instantaneous readings alone
  scored 0.898; the trend mattered more than the instant value.
- P3's R² is moderate, not high, and that's a property of the data, not
  an under-built model: consecutive `RUL_seconds` values in the raw
  data jump up almost as often as they jump down, meaning it behaves
  more like noisy independent samples than a clean countdown.
- P4 does well on the three common fault types but struggles on
  `sensor_fault` (54 of 2,250 rows) — a real, expected limit of a rare
  class with little training data.

## How to add a new idea

Say you want to add an anomaly detector (MAPNA's P5) that needs no
labels at all:

1. Create `models/anomaly_detector.py`, subclass `Predictor`, implement
   `fit`/`predict`/`name`, decorate with
   `@register("model", "anomaly_detector")`.
2. Import the file once in `core/agent.py`'s plugin-import block.
3. Write a `config/agent_config_p5.yaml` pointing at the P5 data, with
   `model.name: anomaly_detector`.
4. Optionally add a matching `DecisionPolicy` if the routing logic
   should differ from the existing ones.

No existing file needs to change — this is the same pattern that took
P2 (binary classification) to P4 (multi-label classification) to P3
(regression) without ever touching `core/agent.py`'s core loop logic
(only its plugin-import list and one line generalizing `target_column`
to accept either a single column or a list).

## Design rationale (for your write-up / pitch)

The reason this is built as a plugin-registry agent rather than three
separate notebooks: predictive maintenance problems vary enormously by
equipment type, available sensors, and failure modes — sometimes you
want "will it fail" (P2), sometimes "what's wrong and who fixes it"
(P4), sometimes "how much time is left" (P3). A founder who can only
ship one hardcoded model has a narrow product. A founder who can plug
in a new equipment profile, a new task type, or a new decision rule in
an afternoon has a platform. This repository's commit history — three
genuinely different ML task types added as clean, incremental branches
on one unchanged core loop — is the actual evidence for that claim, not
just the argument for it.
