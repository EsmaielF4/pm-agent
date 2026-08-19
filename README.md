# Predictive Maintenance Agent — Architecture-First Prototype

## What this is

A failure-prediction **agent**, not just a model. The distinction matters:
a model outputs a probability; an agent perceives, predicts, decides, and
acts. This project is built so accuracy is the least important number in
it — the point is a loop and a plugin system you can keep extending for
four weeks (and beyond) without rewriting anything.

## The loop

```
Perceive → Predict → Decide → Act
 (data)    (models)  (agent)   (output)
```

Every stage is an interface (`base.py` in each folder) with one or more
registered implementations (`core/registry.py`). The `Agent`
(`core/agent.py`) only ever reads `config/agent_config.yaml` and asks the
registry for whatever's named there. It never imports a concrete class
directly.

```
pm_agent/
├── config/agent_config.yaml   <- single control point: which plugins run
├── core/
│   ├── registry.py            <- plugin registration mechanism
│   ├── agent.py                <- orchestrator, runs the loop
│   └── decision.py            <- Decide stage (the actual "agent" reasoning)
├── data/
│   ├── base.py                <- DataSource interface
│   └── sources.py             <- synthetic AI4I generator + CSV loader
├── features/
│   ├── base.py                <- FeatureExtractor interface
│   └── engineering.py
├── models/
│   ├── base.py                <- Predictor interface
│   └── failure_classifier.py  <- baseline Random Forest
├── actions/
│   ├── base.py                <- ActionHandler interface
│   └── alerting.py            <- console + log-file handlers
└── demo/run_demo.py           <- trains + runs one live cycle
```

## Run it

```
cd pm_agent
pip install -r requirements.txt
python demo/run_demo.py
```

## How to add a new idea (the whole point of this architecture)

Say you want to add a Remaining-Useful-Life estimator instead of just a
yes/no failure flag:

1. Create `models/rul_estimator.py`, subclass `Predictor`, implement
   `fit`/`predict`/`name`, decorate with
   `@register("model", "rul_estimator")`.
2. Import the file once in `core/agent.py`'s plugin-import block.
3. Either swap `model.name` in the config to `rul_estimator`, or (better)
   extend `Agent` to run multiple models per cycle and let
   `DecisionPolicy` read both outputs — the interfaces don't stop you
   running several predictors side by side.

Same pattern for a new data source (real sensor feed from equipment
you've worked with), a smarter decision policy (e.g. factoring in time
until the next scheduled maintenance window), or a new action (email,
dashboard, ticketing system). **No existing file needs to change**, only
new files + one config line.

## Update: now running on real MAPNA rotary-equipment data

Switched from the synthetic AI4I (milling-machine/cutting-tool) stand-in
to a real dataset from a separate MAPNA competition, covering **pumps,
turbines, and compressors** — a much closer match to Iran's actual
predictive-maintenance use case than cutting tools.

**What's different about this data, architecturally:** sensors live in
separate CSV files sampled at different rates (1s–15s), so before any
feature/model work happens, readings have to be synchronized onto a
common timeline. That's a new `DataSource` plugin
(`data/multisensor.py`, `mapna_multisensor`) using `pandas.merge_asof`
to align slower sensors onto the label-carrying sensor's timestamps —
everything downstream (feature extractor, model, decision policy,
action handlers) is unchanged, because the interface contract didn't
change, only the plugin.

The MAPNA competition itself is structured as **five sub-problems on
the same kind of sensor data** — which maps cleanly onto this
architecture as five Predictor plugins sharing one data layer:

| Problem | Task | Status |
|---|---|---|
| P2 — pump fault detection | binary classification (`faulted`/`normal`) | **done** — 90.1% acc, 0.899 Macro-F1 on real data |
| P3 — turbine remaining useful life | regression (`RUL_seconds`) | next — reuses `mapna_multisensor` with `already_aligned: true` |
| P4 — fault type + source diagnosis | multi-output classification | next — two classifiers sharing the sensor feature set |
| P5 — anomaly detection | unsupervised, no labels | next — new `Predictor` plugin (e.g. Isolation Forest), same data layer |
| P1 — fixed statistical calculations | deterministic stats script, not really ML | low priority — one-off script, not part of the agent |

Config for the working P2 pipeline: `config/agent_config_mapna_p2.yaml`.
Run it with:
```
python demo/run_demo.py agent_config_mapna_p2.yaml
```

## Current plugins (running on real MAPNA rotary-equipment data)

- **Data**: `mapna_multisensor` (`data/multisensor.py`) — loads real
  pump sensor data from MAPNA's rotary-equipment competition (problem
  P2: fault detection). Handles the core real-world wrinkle this data
  has: each sensor (Temperature, Pressure, VibAccel, VibVelocity) is
  sampled at a different rate (1s/2s/5s/10s) in its own CSV, and the
  label lives only in the slowest sensor's file. This class
  timestamp-aligns everything via `merge_asof` onto the label sensor's
  timeline. It also supports the "pre-aligned" MAPNA layout (P3/P5,
  same row count across files, no timestamp) via `already_aligned:
  true`. The earlier synthetic AI4I generator (`data/sources.py`,
  `synthetic_ai4i`) is kept as a fallback/reference — swapping between
  them is a config change only.
- **Features**: `mapna_sensor_columns` — the 4 raw aligned sensor
  values, NaN-handled (forward/back-fill then median impute, since
  gaps are expected both from the asof-merge and from the raw data
  itself per MAPNA's own documentation).
- **Model**: Random Forest classifier, now label-agnostic (works with
  string classes like `"faulted"/"normal"`, not just 0/1) via a
  `positive_label` config param — needed because scikit-learn's
  `predict_proba` column order follows sorted class names, not which
  one means "bad".
- **Decision policy**: simple probability thresholds (low/medium/high →
  monitor / flag / schedule maintenance). Still the layer to extend
  first — domain judgment plugs in here without retraining anything.
- **Action**: console alert + optional log file.
- **Result**: Macro-F1 ≈ 0.90 on a held-out slice of the real pump
  data — well above the competition's own 0.5 minimum-score threshold,
  with zero hyperparameter tuning.

## The MAPNA dataset (5 sub-problems, all sharing the same plugin slots)

| Problem | Task | Equipment | Sensors | Type | Status |
|---|---|---|---|---|---|
| P2 | Fault detection (normal/faulted) | pump | 4 | Binary classification | **wired up, working** |
| P3 | Remaining Useful Life | turbine | 15 | Regression | data staged, model TODO |
| P4 | Fault type + fault source | rotary equip. | 14 | Multi-label classification | data staged, model TODO |
| P5 | Anomaly detection | rotary equip. | 15 | Unsupervised, no labels | data staged, model TODO |

Each of P3–P5 uses `mapna_multisensor` too (P3/P5 are the
"pre-aligned" layout — set `already_aligned: true`); only a new
`Predictor` plugin (a regressor for P3, a multi-label classifier for
P4, an anomaly detector for P5) is needed to bring each online — this
is the concrete proof that the "expandable" requirement holds in
practice, not just on paper.
