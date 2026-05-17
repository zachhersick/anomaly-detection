# Industrial Anomaly Detection and Alerting Pipeline

An end-to-end machine learning system for simulated industrial sensor anomaly detection.

This project generates synthetic multi-machine sensor data, engineers temporal features, trains an anomaly detection model, evaluates performance, creates row-level alerts, and groups those alerts into higher-level operational events.

The goal is not just to train a model. The goal is to build a realistic ML systems pipeline that connects data generation, feature engineering, model scoring, threshold selection, alerting logic, and operational event grouping.

---

## Project Overview

Industrial equipment often produces continuous sensor readings. A useful anomaly detection system needs to do more than classify rows as normal or abnormal. It also needs to:

- handle time-series behavior
- detect both obvious and subtle anomalies
- separate model suspicion from hard safety violations
- explain why an alert was created
- group noisy row-level alerts into readable operational events
- prepare outputs for storage, APIs, and dashboards

This project currently simulates that full workflow using synthetic industrial sensor data.

---

## Current Pipeline

```text
Synthetic sensor data
        ↓
Temporal feature engineering
        ↓
Random Forest anomaly model
        ↓
Threshold tuning and evaluation
        ↓
Row-level alert generation
        ↓
Grouped alert events
```

Current runnable sequence:

```bash
python generator.py
python features.py
python model.py
python evaluate.py
python alerts.py
python alert_events.py
```

---

## Repository Structure

```text
generator.py          # Generates synthetic multi-machine sensor data
features.py           # Builds temporal/rolling/statistical features
model.py              # Trains model, runs threshold sweep, writes predictions
evaluate.py           # Prints final model evaluation report
alerts.py             # Converts predictions into row-level alerts
alert_events.py       # Groups row-level alerts into operational events
requirements.txt      # Python dependencies
README.md             # Project documentation
```

Generated output files:

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

---

## Data Generation

`generator.py` creates synthetic industrial sensor readings for multiple machines.

### Sensors

```text
temperature
pressure
vibration
flow_rate
voltage
current
```

### Simulated Anomaly Types

```text
spike
drop
drift
oscillation
stuck_sensor
impossible_value
```

### Current Generator Baseline

The generator uses a fixed seed so model performance can be compared consistently across code changes.

```text
fixed_seed = 295
```

Important tuned generator settings include:

```text
current normal noise = 1.0
voltage oscillation noise = 0.85
voltage oscillation amplitude = 0.26
voltage phase-step range = (0.60, 0.66)
```

The generator is currently considered stable enough for the next project phase. Future generator changes should only happen if the generated data is clearly bad or unrealistic.

---

## Feature Engineering

`features.py` reads:

```text
sensor_data_raw.csv
```

and writes:

```text
sensor_data_features.csv
feature_row_retention.csv
```

The feature layer builds temporal features for every sensor, grouped by `machine_id`.

### Feature Categories

```text
delta features
absolute delta features
rolling mean
rolling standard deviation
rolling min/max/range
z-score features
5-step and 10-step differences
rolling slopes
same-direction run length
sign-change counts
lag autocorrelation
centered zero-crossing counts
trend ratio
center balance
```

These features help the model learn time-dependent behavior such as drift, oscillation, stuck sensors, and sudden jumps.

---

## Model Training

`model.py` trains a Random Forest classifier using the engineered features.

### Current Model Setup

```text
model = RandomForestClassifier
class_weight = balanced
DEFAULT_THRESHOLD = 0.35
random_state = 42
test_size = 0.2
```

### Final Feature Decision

The final model keeps all engineered features except the direction-imbalance features.

Permanently dropped suffixes:

```text
_dir_imbalance_10
_dir_imbalance_20
```

These were removed because feature ablation showed that removing them improved or preserved performance while simplifying the model.

---

## Threshold Tuning

The model outputs anomaly probabilities. `model.py` tests multiple classification thresholds:

```text
0.30
0.35
0.40
0.45
0.50
0.55
0.60
0.65
0.70
```

The current selected threshold is:

```text
0.35
```

This threshold was chosen because it gives a strong balance between anomaly recall and false positives.

The script writes:

```text
threshold_results.csv
predictions.csv
```

`predictions.csv` uses the selected default threshold.

---

## Model Evaluation

`evaluate.py` reads:

```text
predictions.csv
feature_row_retention.csv
feature_importance.csv
```

and prints a final evaluation report.

The report includes:

```text
row-retention summary
confusion matrix
accuracy
precision
recall
F1 score
false positives
false negatives
recall by anomaly type
recall by target sensor
drift recall by sensor
oscillation recall by sensor
top feature importances
```

### Current Baseline Performance

Recent selected-threshold behavior:

```text
threshold = 0.35
accuracy ≈ 0.9887
anomaly recall ≈ 0.9749
false positives = 52
false negatives = 59
oscillation recall ≈ 0.9741
```

These numbers may change slightly if the generator, features, model settings, or seed are changed.

---

## Alert Generation

`alerts.py` converts model predictions into row-level alerts.

Input:

```text
predictions.csv
```

Output:

```text
alerts.csv
```

Each row-level alert combines:

```text
model prediction
anomaly score
target sensor
sensor value
safety threshold logic
human-readable reason
```

### Alert Types

```text
model_anomaly
model_and_threshold
```

### Severity Levels

```text
INFO
WARNING
CRITICAL
```

### Alert Logic

A model-only alert is created when the model predicts an anomaly but the sensor value does not cross a hard safety threshold.

A model-and-threshold alert is created when the model predicts an anomaly and the sensor value also violates a warning or critical threshold.

Example reason:

```text
Model predicted anomaly on voltage with high anomaly score 0.812. Sensor value was 118.287 V.
```

Example threshold reason:

```text
pressure value 120.000 PSI exceeded critical high threshold 105.000 PSI.
```

Current row-level alert output:

```text
row-level alerts = 2345
```

---

## Alert Event Grouping

`alert_events.py` groups noisy row-level alerts into higher-level operational events.

Input:

```text
alerts.csv
```

Output:

```text
alert_events.csv
```

This matters because a single real anomaly can create many consecutive row-level alerts. For example, a stuck sensor lasting 80 steps should not appear as 80 separate operational incidents.

### Grouping Rule

Rows are grouped into the same event when they have:

```text
same machine_id
same sensor
same anomaly_type
step gap <= MAX_STEP_GAP
```

Current grouping setting:

```text
MAX_STEP_GAP = 3
```

### Event-Level Fields

Each grouped event tracks:

```text
event_id
machine_id
sensor
anomaly_type
start_step
end_step
duration
alert_count
max_severity
max_severity_reason
max_anomaly_score
mean_anomaly_score
min_sensor_value
max_sensor_value
first_reason
status
real_value
```

### Why `max_severity_reason` Exists

An event can start as a model-only anomaly and later become critical if a later row crosses a hard safety threshold.

Example:

```text
first_reason:
Model predicted anomaly on flow_rate with high anomaly score 0.723. Sensor value was 2.253 m^3/s.

max_severity_reason:
flow_rate value 3.000 m^3/s exceeded critical high threshold 2.750 m^3/s.
```

This preserves both:

```text
why the event started
why the event reached its highest severity
```

Current grouped event output:

```text
row-level alerts = 2345
grouped alert events = 1285
critical grouped events = 30
```

---

## How to Run

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline

```bash
python generator.py
python features.py
python model.py
python evaluate.py
python alerts.py
python alert_events.py
```

### 4. Check generated outputs

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

---

## Requirements

Minimum Python dependencies:

```text
numpy
pandas
scikit-learn
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Current Project Status

Completed:

```text
Synthetic data generator
Temporal feature engineering
Model training
Threshold tuning
Feature ablation
Evaluation script
Row-level alert generation
Grouped alert event generation
README and requirements
```

Current stable baseline:

```text
fixed_seed = 295
DEFAULT_THRESHOLD = 0.35
class_weight = balanced
MAX_STEP_GAP = 3
```

The current system is now ready for the storage/API layer.

---

## Next Planned Work

The next major phase is turning the CSV-based pipeline into a database-backed system.

Planned order:

```text
1. Add SQLite storage layer
2. Load pipeline outputs into database tables
3. Add pipeline run tracking
4. Build FastAPI endpoints
5. Add dashboard/demo layer
```

### Planned Database Tables

```text
sensor_readings
model_predictions
row_alerts
alert_events
pipeline_runs
```

### Planned API Endpoints

```text
GET /health
GET /machines
GET /readings/latest
GET /predictions
GET /alerts
GET /alert-events
GET /alert-events/critical
```

---

## Current Limitations

```text
Synthetic data only
CSV-based storage
No database yet
No API yet
No dashboard yet
No alert acknowledgment workflow yet
No automatic event resolution yet
```

These are intentional current limitations. The next phase addresses storage and API access first.

---

## Intended Project Value

This project demonstrates practical ML systems engineering skills:

```text
synthetic data generation
time-series feature engineering
supervised anomaly detection
threshold selection
model evaluation beyond accuracy
alert severity logic
human-readable alert reasons
event grouping
pipeline design
preparation for database/API/dashboard integration
```

The system is designed to show how model outputs can be turned into operationally useful alerts, not just prediction scores.