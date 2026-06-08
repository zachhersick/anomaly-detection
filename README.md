# Industrial Anomaly Detection and Alerting Platform

![CI](https://github.com/zachhersick/anomaly-detection/actions/workflows/ci.yml/badge.svg)

An end-to-end ML systems project for detecting and visualizing simulated industrial machine anomalies.

The project generates synthetic sensor data, engineers time-series features, trains a machine learning model, creates alerts, groups alerts into operational events, stores results in SQLite, exposes results through a FastAPI backend, and displays them in a Streamlit dashboard.

```text
Synthetic data
    ↓
Feature engineering
    ↓
Model training
    ↓
Prediction + thresholding
    ↓
Row-level alerts
    ↓
Grouped alert events
    ↓
SQLite storage
    ↓
FastAPI backend
    ↓
Streamlit dashboard
```

---

## Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the ML pipeline

```bash
python run_pipeline.py
```

This creates the generated CSV outputs.

### 3. Load outputs into SQLite

```bash
python load_to_db.py
```

### 4. Start the API

```bash
python -m uvicorn api:app --reload
```

### 5. Start the dashboard

```bash
streamlit run dashboard.py
```

---

## Project Goals

This project is designed to demonstrate a realistic ML systems workflow, not just a notebook model.

It focuses on:

```text
synthetic data generation
time-series feature engineering
model training and evaluation
alerting logic
event grouping
database persistence
REST API design
dashboard-ready endpoints
basic frontend visualization
testing and CI
```

---

## Tech Stack

```text
Python
pandas
NumPy
scikit-learn
SQLite
FastAPI
Pydantic
Uvicorn
Streamlit
pytest
GitHub Actions
```

---

## Repository Structure

```text
generator.py              Generate synthetic industrial sensor data
features.py               Create temporal/statistical features
model.py                  Train model and write predictions
evaluate.py               Print evaluation report
alerts.py                 Create row-level alerts
alert_events.py           Group row-level alerts into alert events
run_pipeline.py           Run the full ML pipeline

db.py                     SQLite schema, connection, and indexes
load_to_db.py             Load generated outputs into SQLite
db_queries.py             Read/query helpers for the API

api.py                    FastAPI backend
schemas.py                Pydantic response models
dashboard.py              Streamlit dashboard client

tests/                    Pytest suite
.github/workflows/ci.yml  GitHub Actions CI
requirements.txt          Python dependencies
```

---

## Synthetic Data

The generator simulates multiple industrial machines with the following sensors:

```text
temperature
pressure
vibration
flow_rate
voltage
current
```

Supported anomaly types:

```text
spike
drop
drift
oscillation
stuck_sensor
impossible_value
```

The generated dataset includes normal behavior, noisy sensor readings, and labeled anomaly periods.

Output:

```text
sensor_data_raw.csv
```

---

## Feature Engineering

`features.py` creates time-series features per machine and sensor.

Feature examples:

```text
rolling mean
rolling standard deviation
rolling min/max/range
delta features
absolute delta features
z-scores
short-vs-long rolling mean differences
rolling slope
same-direction run length
centered zero-crossing count
lag autocorrelation
```

Outputs:

```text
sensor_data_features.csv
feature_row_retention.csv
```

---

## Model Training

`model.py` trains a Random Forest classifier to predict whether a row is anomalous.

The model writes:

```text
predictions.csv
feature_importance.csv
outputs/threshold_results.csv
```

The project currently uses a selected anomaly threshold of:

```text
0.35
```

---

## Alerting

`alerts.py` converts model predictions into row-level alerts.

Each alert includes:

```text
machine_id
step
sensor
sensor_value
prediction
anomaly_score
severity
alert_type
reason
status
anomaly_type
```

Severity levels:

```text
INFO
WARNING
CRITICAL
```

Output:

```text
alerts.csv
```

---

## Alert Event Grouping

`alert_events.py` groups consecutive row-level alerts into higher-level operational events.

This prevents one real anomaly from appearing as many unrelated alerts.

Grouping is based on:

```text
machine_id
sensor
anomaly_type
step gap
```

Output:

```text
alert_events.csv
```

Each event tracks:

```text
start_step
end_step
duration
alert_count
max_severity
max_anomaly_score
mean_anomaly_score
sensor value range
status
```

---

## SQLite Storage

The project stores pipeline outputs in SQLite.

Database file:

```text
anomaly_detection.db
```

Tables:

```text
pipeline_runs
sensor_readings
model_predictions
row_alerts
alert_events
```

Load generated pipeline outputs into SQLite:

```bash
python load_to_db.py
```

Each load creates a new `run_id`, allowing multiple pipeline runs to be stored and compared.

The database also includes indexes for common API query paths, including event filtering, prediction filtering, and sensor reading pagination.

---

## FastAPI Backend

Start the API server:

```bash
python -m uvicorn api:app --reload
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

Main endpoints:

```text
GET /health
GET /runs
GET /runs/latest

GET /runs/{run_id}/summary

GET /runs/{run_id}/events
GET /runs/{run_id}/events/critical
GET /runs/{run_id}/events/{event_id}/alerts

GET /runs/{run_id}/events/anomaly-type-distribution
GET /runs/{run_id}/events/sensor-distribution
GET /runs/{run_id}/events/severity-distribution

GET /runs/{run_id}/predictions
GET /runs/{run_id}/machines/{machine_id}/readings

GET /dashboard/runs/{run_id}
```

---

## API Features

The API supports:

```text
run validation
event validation
filtering
pagination
response models
dashboard-ready aggregate responses
```

Filtering examples:

```text
GET /runs/1/events?severity=CRITICAL
GET /runs/1/events?sensor=temperature
GET /runs/1/events?anomaly_type=spike
GET /runs/1/predictions?machine_id=1
GET /runs/1/predictions?target_sensor=temperature
```

Pagination examples:

```text
GET /runs/1/events?limit=100&offset=0
GET /runs/1/predictions?limit=100&offset=100
GET /runs/1/machines/1/readings?limit=100&offset=200
```

Pagination rules:

```text
limit: 1 to 500
offset: 0 or greater
```

Invalid values return FastAPI validation errors.

---

## Dashboard Endpoint

The main dashboard endpoint is:

```text
GET /dashboard/runs/{run_id}
```

It returns:

```text
summary metrics
anomaly type distribution
sensor distribution
severity distribution
top critical events
```

This lets the dashboard load the main run overview with one API request.

---

## Streamlit Dashboard

The project includes a Streamlit dashboard that calls the FastAPI backend.

Start the API first:

```bash
python -m uvicorn api:app --reload
```

Then start the dashboard in a second terminal:

```bash
streamlit run dashboard.py
```

The dashboard displays:

```text
run summary metrics
anomaly type distribution chart
sensor distribution chart
severity distribution chart
top critical events table
raw API response for debugging
```

The dashboard does not read CSV files or SQLite directly. It uses FastAPI as the data access layer.

```text
Streamlit -> FastAPI -> SQLite
```

---

## Testing

Run all tests:

```bash
python -m pytest
```

Run API tests only:

```bash
python -m pytest tests/test_api.py
```

Run database tests only:

```bash
python -m pytest tests/test_db.py
```

The test suite covers:

```text
data generation
feature engineering
model helpers
alert creation
alert event grouping
database schema
database loading
database indexes
query helpers
FastAPI routes
filtering
pagination
validation
summary endpoint
distribution endpoints
dashboard endpoint
pipeline orchestration
```

---

## Continuous Integration

GitHub Actions runs the test suite on:

```text
push
pull_request
```

Workflow file:

```text
.github/workflows/ci.yml
```

---

## Generated Files

The following files are generated locally and should not be committed:

```text
*.csv
*.db
*.sqlite
*.sqlite3
metrics.txt
outputs/threshold_results.csv
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

---

## Current Status

Completed:

```text
synthetic data generator
feature engineering
Random Forest anomaly model
threshold sweep
evaluation script
row-level alerting
alert event grouping
SQLite persistence
SQLite indexes
FastAPI backend
API filtering and pagination
API validation
dashboard-ready API endpoint
Streamlit dashboard
pytest test suite
GitHub Actions CI
```

Planned final polish:

```text
model ablation script
dashboard screenshots
final README screenshots/demo section
optional Docker support
```

---

## Project Summary

This project demonstrates the full path from ML model output to operational application behavior.

It includes the backend systems work around the model:

```text
data generation
features
modeling
alerts
events
storage
API access
dashboard visualization
tests
CI
```

The result is a portfolio-ready ML systems project that shows more than model training alone.