# Industrial Anomaly Detection and Alerting Pipeline

This project is an end-to-end machine learning system for simulated industrial sensor anomaly detection. It generates synthetic multi-machine sensor data, engineers temporal features, trains an anomaly detection model, evaluates performance, creates row-level alerts, and groups those alerts into higher-level operational events.

The main goal is to show the full ML systems workflow, not just model training. The project connects data simulation, feature engineering, model evaluation, alert logic, and event grouping into one runnable pipeline.

## Current pipeline

```text
sensor simulation
-> temporal feature engineering
-> model training and threshold tuning
-> model evaluation
-> row-level alert generation
-> grouped alert events
```

## Repository scripts

### `generator.py`

Generates synthetic industrial sensor data for multiple machines.

Sensors:

- temperature
- pressure
- vibration
- flow_rate
- voltage
- current

Anomaly types:

- spike
- drop
- drift
- oscillation
- stuck_sensor
- impossible_value

The generator currently uses a fixed seed so model and alert performance can be compared consistently across code changes.

### `features.py`

Reads `sensor_data_raw.csv` and creates engineered temporal features for each sensor.

Feature categories include:

- deltas and absolute deltas
- rolling means and standard deviations
- rolling ranges
- z-scores
- multi-step differences
- slopes
- same-direction run length
- sign-change counts
- lag autocorrelation
- centered zero-crossing counts
- trend ratio
- center balance

The script writes:

```text
sensor_data_features.csv
feature_row_retention.csv
```

### `model.py`

Trains a Random Forest classifier on the engineered features.

Current model choices:

- model: `RandomForestClassifier`
- class weighting: `class_weight='balanced'`
- train/test split: stratified
- default threshold: `0.35`
- permanently dropped feature suffixes:
  - `_dir_imbalance_10`
  - `_dir_imbalance_20`

The script performs a threshold sweep and writes:

```text
predictions.csv
predictions_threshold_0_3.csv
predictions_threshold_0_35.csv
predictions_threshold_0_4.csv
predictions_threshold_0_45.csv
predictions_threshold_0_5.csv
predictions_threshold_0_55.csv
predictions_threshold_0_6.csv
predictions_threshold_0_65.csv
predictions_threshold_0_7.csv
threshold_results.csv
feature_importance.csv
```

### `evaluate.py`

Reads `predictions.csv` and prints a final model evaluation report.

The report includes:

- row-retention summary
- confusion matrix
- accuracy
- precision
- recall
- F1 score
- false positives
- false negatives
- recall by anomaly type
- recall by target sensor
- drift recall by sensor
- oscillation recall by sensor
- top feature importances

### `alerts.py`

Reads `predictions.csv` and creates row-level alerts for every row where the model predicts an anomaly.

Alert types:

- `model_anomaly`
- `model_and_threshold`

Severity levels:

- `INFO`
- `WARNING`
- `CRITICAL`

The alerting layer combines:

```text
model prediction + anomaly score + target sensor + sensor value + safety threshold logic
```

The script writes:

```text
alerts.csv
```

### `alert_events.py`

Reads `alerts.csv` and groups row-level alerts into higher-level operational alert events.

Rows are grouped into the same event when they have:

- the same `machine_id`
- the same `sensor`
- the same `anomaly_type`
- a step gap less than or equal to `MAX_STEP_GAP`

Current setting:

```text
MAX_STEP_GAP = 3
```

The script tracks:

- start and end step
- duration
- alert count
- max severity
- max severity reason
- max anomaly score
- mean anomaly score
- min and max sensor values
- first reason
- status

The script writes:

```text
alert_events.csv
```

## How to run the full pipeline

From the repository root:

```bash
python generator.py
python features.py
python model.py
python evaluate.py
python alerts.py
python alert_events.py
```

Expected output files:

```text
sensor_data_raw.csv
sensor_data_features.csv
feature_row_retention.csv
predictions.csv
threshold_results.csv
feature_importance.csv
alerts.csv
alert_events.csv
```

## Current baseline

The current selected model setup uses:

```text
fixed_seed = 295
DEFAULT_THRESHOLD = 0.35
class_weight = 'balanced'
MAX_STEP_GAP = 3
```

Recent pipeline behavior:

```text
row-level alerts: 2345
grouped alert events: 1285
critical grouped events: 30
```

The model currently catches most injected anomalies while keeping false positives relatively low. The alert layer then separates model-only anomalies from physically unsafe threshold violations.

## Why alert events matter

Row-level alerts are useful for debugging, but they are too noisy for operations. A single stuck sensor or oscillation can produce many consecutive anomalous rows. `alert_events.py` compresses these row-level alerts into event-level summaries that are closer to what an operator would actually review.

Example:

```text
Before grouping:
step 100, machine 3, voltage, oscillation
step 101, machine 3, voltage, oscillation
step 102, machine 3, voltage, oscillation

After grouping:
machine 3, voltage, oscillation, start_step 100, end_step 102, duration 3
```

## Current limitations

- The data is synthetic, not from real industrial equipment.
- The system currently writes intermediate CSV files rather than using a database.
- The model is trained and evaluated in script form, not inside a production training framework.
- There is no API layer yet.
- There is no dashboard yet.
- Alert acknowledgment and resolution workflows are not implemented yet.

## Next planned work

The next major stage is the storage/API layer.

Planned order:

1. Add a SQLite storage layer.
2. Load pipeline outputs into database tables.
3. Add a `pipeline_runs` table to track seed, threshold, row counts, and run metadata.
4. Build FastAPI endpoints for readings, predictions, alerts, and grouped events.
5. Add a simple dashboard after the API is working.

Suggested first database tables:

```text
sensor_readings
model_predictions
row_alerts
alert_events
pipeline_runs
```

## Intended project value

This project is designed to demonstrate practical ML systems skills:

- generating and validating synthetic data
- engineering temporal features
- training and tuning a classifier
- evaluating anomaly detection performance beyond simple accuracy
- designing alert logic around model outputs and safety thresholds
- grouping noisy model outputs into operational events
- preparing the system for database-backed APIs and dashboards
