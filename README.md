# Industrial Anomaly Detection and Alerting Platform

![CI](https://github.com/zachhersick/anomaly-detection/actions/workflows/ci.yml/badge.svg)

An end-to-end machine learning systems project for simulated industrial anomaly detection.

This project generates synthetic multi-machine sensor data, engineers temporal features, trains an anomaly detection model, evaluates performance, creates row-level alerts, groups alerts into operational events, stores results in SQLite, and exposes the stored results through a FastAPI REST API.

The goal is not just to train a model. The goal is to build a realistic ML systems pipeline that connects:

```text
data generation
    ↓
feature engineering
    ↓
model training and thresholding
    ↓
alert generation
    ↓
event grouping
    ↓
SQLite persistence
    ↓
FastAPI read API
    ↓
future dashboard or deployment
```

---

## Project Overview

Industrial systems produce continuous sensor readings. A useful anomaly detection platform needs to do more than classify rows as normal or anomalous. It should also:

- model time-series behavior
- detect multiple anomaly patterns
- score anomalies with a model
- apply safety threshold logic
- explain why alerts were created
- group noisy row-level alerts into operational events
- persist pipeline outputs for later analysis
- expose results through an API

This project simulates that workflow using synthetic industrial sensor data.

---

## Current Architecture

```text
generator.py
    ↓
sensor_data_raw.csv

features.py
    ↓
sensor_data_features.csv
    ↓
feature_row_retention.csv

model.py
    ↓
predictions.csv
    ↓
outputs/threshold_results.csv
    ↓
feature_importance.csv

evaluate.py
    ↓
console evaluation report

alerts.py
    ↓
alerts.csv

alert_events.py
    ↓
alert_events.csv

load_to_db.py
    ↓
anomaly_detection.db

db_queries.py
    ↓
read/query helpers

api.py
    ↓
FastAPI REST endpoints
```

The full script pipeline can be run with:

```bash
python run_pipeline.py
```

`run_pipeline.py` runs the scripts in this order:

```text
generator.py
features.py
model.py
evaluate.py
alerts.py
alert_events.py
```

After the CSV outputs are created, they can be loaded into SQLite with:

```bash
python load_to_db.py
```

---

## Repository Structure

```text
api.py                     FastAPI application and API routes
schemas.py                 Pydantic response models for API responses

db.py                      SQLite connection, table creation, and indexes
db_queries.py              Read-only SQLite query helpers
load_to_db.py              Loads CSV pipeline outputs into SQLite

run_pipeline.py            Runs the main ML pipeline scripts in order
generator.py               Generates synthetic industrial sensor data
features.py                Builds temporal and statistical features
model.py                   Trains model, runs threshold sweep, writes predictions
evaluate.py                Prints model evaluation report
alerts.py                  Converts predictions into row-level alerts
alert_events.py            Groups row-level alerts into event-level incidents

tests/                     Pytest test suite
.github/workflows/ci.yml   GitHub Actions CI workflow

requirements.txt           Python dependencies
README.md                  Project documentation
```

---

## Synthetic Data Generation

`generator.py` creates synthetic industrial sensor data for multiple machines.

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

### Generator Details

The generator currently uses:

```text
num_machines = 10
num_timesteps = 5000
fixed_seed = 295
```

This produces:

```text
50,000 raw sensor rows
```

The generator models each machine independently, tracks per-machine state, and applies anomalies to a target sensor for a selected duration.

Output:

```text
sensor_data_raw.csv
```

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

Features are built per machine and per sensor.

### Feature Categories

```text
delta
absolute delta
rolling mean
rolling standard deviation
rolling min/max/range
z-score
5-step and 10-step difference
same-direction run length
sign-change count
short-vs-long rolling mean difference
rolling slope
cumulative change
long-baseline deviation
lag autocorrelation
centered zero-crossing count
trend ratio
center balance
```

These features are meant to help the model detect both obvious and temporal anomalies, including drift, oscillation, stuck sensors, and sudden jumps.

---

## Model Training

`model.py` trains a Random Forest classifier on the engineered features.

Current model setup:

```text
RandomForestClassifier
n_estimators = 300
class_weight = balanced
random_state = 42
test_size = 0.2
```

The model predicts whether a row is anomalous.

The model outputs anomaly probabilities, then applies classification thresholds.

### Threshold Sweep

The model currently evaluates:

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

The selected default threshold is:

```text
0.35
```

Outputs:

```text
predictions.csv
outputs/threshold_results.csv
feature_importance.csv
```

`predictions.csv` is written using the selected default threshold.

---

## Model Evaluation

`evaluate.py` reads model outputs and prints an evaluation report.

The evaluation includes:

```text
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
feature-row retention
```

This is used to inspect both overall model performance and weak spots by anomaly type.

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

Each alert combines:

```text
model prediction
anomaly score
target sensor
sensor value
safety threshold logic
severity
alert type
human-readable reason
status
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

Model-only alerts are created when the model predicts an anomaly but the sensor value does not cross a hard safety threshold.

Model-and-threshold alerts are created when the model predicts an anomaly and the sensor value also violates a warning or critical threshold.

---

## Alert Event Grouping

`alert_events.py` groups row-level alerts into higher-level operational events.

Input:

```text
alerts.csv
```

Output:

```text
alert_events.csv
```

This matters because one real anomaly can trigger many consecutive row-level alerts. For example, a stuck sensor lasting 80 steps should be treated as one operational event, not 80 unrelated incidents.

### Grouping Rule

Alerts are grouped into the same event when they share:

```text
machine_id
sensor
anomaly_type
step gap <= MAX_STEP_GAP
```

Current setting:

```text
MAX_STEP_GAP = 3
```

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

---

## SQLite Storage

The project includes a SQLite persistence layer.

### Database Setup

`db.py` creates:

```text
pipeline_runs
sensor_readings
model_predictions
row_alerts
alert_events
```

The database file is:

```text
anomaly_detection.db
```

This file is generated locally and should not be committed.

### Load Pipeline Outputs into SQLite

After running the pipeline, load the CSV outputs into SQLite:

```bash
python load_to_db.py
```

This loads:

```text
sensor_data_raw.csv  -> sensor_readings
predictions.csv      -> model_predictions
alerts.csv           -> row_alerts
alert_events.csv     -> alert_events
```

Each load creates a new row in:

```text
pipeline_runs
```

and attaches the same `run_id` to all inserted rows. This makes it possible to compare multiple pipeline runs over time.

### SQLite Indexes

`db.py` also creates indexes for common API query paths.

The indexes support:

```text
alert event lookup by run_id and start_step
alert event filtering by severity, sensor, and anomaly_type
prediction lookup by run_id, machine_id, anomaly_type, target_sensor, and step
sensor reading lookup by run_id, machine_id, and step
```

These indexes are important because the API supports filtering and pagination. Without indexes, SQLite may need to scan entire tables as stored pipeline runs grow.

---

## FastAPI REST API

The project exposes stored results through a FastAPI API.

Start the API server:

```bash
python -m uvicorn api:app --reload
```

Open the interactive docs:

```text
http://127.0.0.1:8000/docs
```

### API Endpoints

```text
GET /health
GET /runs
GET /runs/latest

GET /runs/{run_id}/events
GET /runs/{run_id}/events/critical
GET /runs/{run_id}/events/{event_id}/alerts

GET /runs/{run_id}/machines/{machine_id}/readings
GET /runs/{run_id}/predictions
```

### Event Filtering and Pagination

`GET /runs/{run_id}/events` supports:

```text
severity
sensor
anomaly_type
limit
offset
```

Example requests:

```text
http://127.0.0.1:8000/runs/1/events
http://127.0.0.1:8000/runs/1/events?severity=CRITICAL
http://127.0.0.1:8000/runs/1/events?sensor=temperature
http://127.0.0.1:8000/runs/1/events?anomaly_type=spike
http://127.0.0.1:8000/runs/1/events?severity=CRITICAL&sensor=temperature&limit=100&offset=0
```

Results are ordered by:

```text
start_step ASC
```

### Prediction Filtering and Pagination

`GET /runs/{run_id}/predictions` supports:

```text
machine_id
anomaly_type
target_sensor
limit
offset
```

Example requests:

```text
http://127.0.0.1:8000/runs/1/predictions
http://127.0.0.1:8000/runs/1/predictions?machine_id=1
http://127.0.0.1:8000/runs/1/predictions?anomaly_type=spike
http://127.0.0.1:8000/runs/1/predictions?target_sensor=temperature
http://127.0.0.1:8000/runs/1/predictions?machine_id=1&target_sensor=temperature&limit=100&offset=0
```

Results are ordered by:

```text
step ASC, machine_id ASC, target_sensor ASC
```

### Sensor Reading Pagination

`GET /runs/{run_id}/machines/{machine_id}/readings` supports:

```text
limit
offset
```

Example requests:

```text
http://127.0.0.1:8000/runs/1/machines/1/readings
http://127.0.0.1:8000/runs/1/machines/1/readings?limit=100
http://127.0.0.1:8000/runs/1/machines/1/readings?limit=100&offset=100
```

Results are ordered by:

```text
step ASC
```

### Pagination Validation

For paginated endpoints:

```text
limit must be between 1 and 500
offset must be greater than or equal to 0
```

Invalid pagination values return a FastAPI validation error:

```text
422 Validation Error
```

Examples of invalid requests:

```text
/runs/1/events?limit=0
/runs/1/events?limit=501
/runs/1/events?offset=-1

/runs/1/predictions?limit=0
/runs/1/predictions?limit=501
/runs/1/predictions?offset=-1

/runs/1/machines/1/readings?limit=0
/runs/1/machines/1/readings?limit=501
/runs/1/machines/1/readings?offset=-1
```

### Response Models

`schemas.py` defines Pydantic response models for:

```text
HealthResponse
PipelineRunResponse
LatestRunResponse
AlertEventResponse
RowAlertResponse
SensorReadingResponse
PredictionResponse
```

These models make the API contract explicit and improve the generated `/docs` page.

### Error Handling

Run-specific endpoints validate that the requested `run_id` exists.

If the run does not exist, the API returns:

```text
404 Pipeline run not found
```

Event-specific endpoints validate that the requested `event_id` exists for the selected run.

If the event does not exist, the API returns:

```text
404 Alert event not found
```

---

## Testing

The project uses `pytest`.

Run all tests:

```bash
python -m pytest
```

Run only API tests:

```bash
python -m pytest tests/test_api.py
```

Run only database tests:

```bash
python -m pytest tests/test_db.py
```

The test suite covers:

```text
synthetic data generation
feature engineering
model training helpers
evaluation helpers
alert generation
alert event grouping
SQLite table creation/loading
SQLite indexes
SQLite query helpers
FastAPI endpoints
API filtering
API pagination
API query validation
API response models
missing-run validation
missing-event validation
pipeline orchestration
```

---

## Continuous Integration

GitHub Actions is configured in:

```text
.github/workflows/ci.yml
```

The workflow runs on:

```text
push
pull_request
```

It:

```text
checks out the repository
sets up Python 3.11
installs requirements.txt
runs python -m pytest
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/zachhersick/anomaly-detection.git
cd anomaly-detection
```

### 2. Create a virtual environment

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Run the full ML pipeline

```bash
python run_pipeline.py
```

This creates the CSV outputs.

### Load outputs into SQLite

```bash
python load_to_db.py
```

This creates or updates:

```text
anomaly_detection.db
```

### Start the API

```bash
python -m uvicorn api:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

### Run tests

```bash
python -m pytest
```

---

## Generated Files

These files are generated locally and ignored by Git:

```text
*.csv
anomaly_detection.db
*.db
*.sqlite
*.sqlite3
metrics.txt
.pytest_cache/
__pycache__/
```

Common generated outputs:

```text
sensor_data_raw.csv
sensor_data_features.csv
feature_row_retention.csv
predictions.csv
feature_importance.csv
alerts.csv
alert_events.csv
anomaly_detection.db
```

The threshold sweep output is stored under:

```text
outputs/threshold_results.csv
```

Generated files should not be committed.

---

## Current Project Status

Completed:

```text
Synthetic multi-machine data generator
Temporal feature engineering
Random Forest anomaly model
Threshold sweep
Evaluation script
Row-level alert generation
Grouped alert event generation
SQLite schema and persistence layer
SQLite query helpers
SQLite indexes for common API query paths
FastAPI read API
Pydantic response models
API missing-run validation
API missing-event validation
Event filtering
Prediction filtering
Pagination with limit and offset
Pagination validation
Pytest coverage
GitHub Actions CI
MIT license
```

Next planned improvements:

```text
Add run summary endpoint for dashboard usage
Add dashboard-oriented aggregate queries
Persist feature rows if needed
Add deployment configuration
Build a simple dashboard or frontend
```

---

## Technical Stack

```text
Python
pandas
NumPy
scikit-learn
SQLite
FastAPI
Pydantic
Uvicorn
pytest
GitHub Actions
```

---

## Notes

This project is intentionally structured as an ML systems project rather than a standalone notebook. The pipeline is designed to show the full path from data generation to operational API access.

The main engineering focus is:

```text
reproducible synthetic data
time-aware feature engineering
model thresholding
alert/event logic
persistent storage
REST API design
database indexing
test coverage
CI
```