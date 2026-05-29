import sqlite3

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import app, get_db_connection
from db import create_tables
from load_to_db import insert_pipeline_run, load_dataframe_to_table


@pytest.fixture
def temp_connection(tmp_path):
    """
    Create a temporary SQLite database for each test.
    """
    db_path = tmp_path / "test_anomaly_detection.db"

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")

    create_tables(conn)

    yield conn

    conn.close()


@pytest.fixture
def client(temp_connection):
    """
    Create a FastAPI test client that uses the temporary test database.
    """

    def test_get_db_connection():
        yield temp_connection

    app.dependency_overrides[get_db_connection] = test_get_db_connection

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_read_pipeline_runs(client, temp_connection):
    older_run_id = insert_pipeline_run(
        temp_connection,
        notes="older test run",
        fixed_seed=42,
        model_threshold=0.80,
        max_step_gap=3,
    )

    newer_run_id = insert_pipeline_run(
        temp_connection,
        notes="newer test run",
        fixed_seed=99,
        model_threshold=0.90,
        max_step_gap=5,
    )

    response = client.get("/runs")

    assert response.status_code == 200

    data = response.json()
    returned_run_ids = [run["run_id"] for run in data]

    assert returned_run_ids == [newer_run_id, older_run_id]


def test_read_latest_run(client, temp_connection):
    older_run_id = insert_pipeline_run(
        temp_connection,
        notes="older test run",
        fixed_seed=42,
        model_threshold=0.80,
        max_step_gap=3,
    )

    newer_run_id = insert_pipeline_run(
        temp_connection,
        notes="newer test run",
        fixed_seed=99,
        model_threshold=0.90,
        max_step_gap=5,
    )

    response = client.get("/runs/latest")

    assert response.status_code == 200
    assert response.json() == {"run_id": newer_run_id}
    assert response.json()["run_id"] > older_run_id


def test_read_latest_run_returns_404_when_no_runs_exist(client):
    response = client.get("/runs/latest")

    assert response.status_code == 404
    assert response.json()["detail"] == "No pipeline runs found."


def test_read_alert_events_for_run(client, temp_connection):
    run_id_1 = insert_pipeline_run(temp_connection, notes="run one")
    run_id_2 = insert_pipeline_run(temp_connection, notes="run two")

    run_1_events = pd.DataFrame(
        [
            {
                "event_id": 1,
                "machine_id": 1,
                "sensor": "temperature",
                "anomaly_type": "spike",
                "start_step": 20,
                "end_step": 25,
                "duration": 6,
                "alert_count": 3,
                "max_severity": "WARNING",
                "status": "open",
            },
            {
                "event_id": 2,
                "machine_id": 1,
                "sensor": "pressure",
                "anomaly_type": "drop",
                "start_step": 10,
                "end_step": 12,
                "duration": 3,
                "alert_count": 2,
                "max_severity": "CRITICAL",
                "status": "open",
            },
        ]
    )

    run_2_events = pd.DataFrame(
        [
            {
                "event_id": 1,
                "machine_id": 2,
                "sensor": "voltage",
                "anomaly_type": "drift",
                "start_step": 5,
                "end_step": 15,
                "duration": 11,
                "alert_count": 4,
                "max_severity": "WARNING",
                "status": "open",
            }
        ]
    )

    load_dataframe_to_table(temp_connection, run_1_events, "alert_events", run_id_1)
    load_dataframe_to_table(temp_connection, run_2_events, "alert_events", run_id_2)

    response = client.get(f"/runs/{run_id_1}/events")

    assert response.status_code == 200

    data = response.json()
    returned_event_ids = [event["event_id"] for event in data]

    assert len(data) == 2
    assert returned_event_ids == [2, 1]
    assert all(event["run_id"] == run_id_1 for event in data)


def test_read_critical_alert_events_for_run(client, temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    events = pd.DataFrame(
        [
            {
                "event_id": 1,
                "machine_id": 1,
                "sensor": "temperature",
                "anomaly_type": "spike",
                "start_step": 10,
                "end_step": 12,
                "duration": 3,
                "alert_count": 2,
                "max_severity": "WARNING",
                "status": "open",
            },
            {
                "event_id": 2,
                "machine_id": 1,
                "sensor": "pressure",
                "anomaly_type": "drop",
                "start_step": 20,
                "end_step": 22,
                "duration": 3,
                "alert_count": 2,
                "max_severity": "CRITICAL",
                "status": "open",
            },
        ]
    )

    load_dataframe_to_table(temp_connection, events, "alert_events", run_id)

    response = client.get(f"/runs/{run_id}/events/critical")

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["event_id"] == 2
    assert data[0]["max_severity"] == "CRITICAL"
    
def test_read_row_alerts_for_event(client, temp_connection):
    run_id = insert_pipeline_run(temp_connection)

    events = pd.DataFrame(
        [
            {
                "event_id": 1,
                "machine_id": 1,
                "sensor": "temperature",
                "anomaly_type": "spike",
                "start_step": 10,
                "end_step": 12,
                "duration": 3,
                "alert_count": 3,
                "max_severity": "CRITICAL",
                "status": "open",
            }
        ]
    )

    alerts = pd.DataFrame(
        [
            {
                "alert_id": 1,
                "step": 9,
                "machine_id": 1,
                "sensor": "temperature",
                "sensor_value": 99.0,
                "prediction": 1,
                "anomaly_score": 0.70,
                "severity": "WARNING",
                "alert_type": "model_anomaly",
                "reason": "Before event window",
                "status": "open",
                "anomaly_type": "spike",
                "real_value": 1,
            },
            {
                "alert_id": 2,
                "step": 10,
                "machine_id": 1,
                "sensor": "temperature",
                "sensor_value": 105.0,
                "prediction": 1,
                "anomaly_score": 0.90,
                "severity": "CRITICAL",
                "alert_type": "model_and_threshold",
                "reason": "Inside event window",
                "status": "open",
                "anomaly_type": "spike",
                "real_value": 1,
            },
            {
                "alert_id": 3,
                "step": 11,
                "machine_id": 1,
                "sensor": "temperature",
                "sensor_value": 106.0,
                "prediction": 1,
                "anomaly_score": 0.91,
                "severity": "CRITICAL",
                "alert_type": "model_and_threshold",
                "reason": "Inside event window",
                "status": "open",
                "anomaly_type": "spike",
                "real_value": 1,
            },
            {
                "alert_id": 4,
                "step": 12,
                "machine_id": 1,
                "sensor": "temperature",
                "sensor_value": 107.0,
                "prediction": 1,
                "anomaly_score": 0.92,
                "severity": "CRITICAL",
                "alert_type": "model_and_threshold",
                "reason": "Inside event window",
                "status": "open",
                "anomaly_type": "spike",
                "real_value": 1,
            },
            {
                "alert_id": 5,
                "step": 13,
                "machine_id": 1,
                "sensor": "temperature",
                "sensor_value": 101.0,
                "prediction": 1,
                "anomaly_score": 0.72,
                "severity": "WARNING",
                "alert_type": "model_anomaly",
                "reason": "After event window",
                "status": "open",
                "anomaly_type": "spike",
                "real_value": 1,
            },
        ]
    )

    load_dataframe_to_table(temp_connection, events, "alert_events", run_id)
    load_dataframe_to_table(temp_connection, alerts, "row_alerts", run_id)

    response = client.get(f"/runs/{run_id}/events/1/alerts")

    assert response.status_code == 200

    data = response.json()
    returned_steps = [alert["step"] for alert in data]

    assert returned_steps == [10, 11, 12]
    assert all(alert["run_id"] == run_id for alert in data)
    assert all(alert["machine_id"] == 1 for alert in data)
    assert all(alert["sensor"] == "temperature" for alert in data)
    
def test_read_sensor_readings_for_machine(client, temp_connection):
    run_id_1 = insert_pipeline_run(temp_connection)
    run_id_2 = insert_pipeline_run(temp_connection)

    run_1_readings = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01 00:00:01",
                "step": 1,
                "machine_id": 1,
                "temperature": 70.0,
                "pressure": 50.0,
                "vibration": 2.0,
                "flow_rate": 1.0,
                "voltage": 120.0,
                "current": 10.0,
                "is_anomaly": 0,
                "anomaly_type": "normal",
                "target_sensor": "temperature",
            },
            {
                "timestamp": "2026-01-01 00:00:02",
                "step": 2,
                "machine_id": 1,
                "temperature": 71.0,
                "pressure": 51.0,
                "vibration": 2.1,
                "flow_rate": 1.1,
                "voltage": 121.0,
                "current": 11.0,
                "is_anomaly": 0,
                "anomaly_type": "normal",
                "target_sensor": "temperature",
            },
            {
                "timestamp": "2026-01-01 00:00:03",
                "step": 3,
                "machine_id": 1,
                "temperature": 72.0,
                "pressure": 52.0,
                "vibration": 2.2,
                "flow_rate": 1.2,
                "voltage": 122.0,
                "current": 12.0,
                "is_anomaly": 0,
                "anomaly_type": "normal",
                "target_sensor": "temperature",
            },
            {
                "timestamp": "2026-01-01 00:00:01",
                "step": 1,
                "machine_id": 2,
                "temperature": 80.0,
                "pressure": 60.0,
                "vibration": 3.0,
                "flow_rate": 1.5,
                "voltage": 125.0,
                "current": 15.0,
                "is_anomaly": 0,
                "anomaly_type": "normal",
                "target_sensor": "temperature",
            },
        ]
    )

    run_2_readings = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01 00:00:01",
                "step": 1,
                "machine_id": 1,
                "temperature": 90.0,
                "pressure": 70.0,
                "vibration": 4.0,
                "flow_rate": 1.8,
                "voltage": 127.0,
                "current": 17.0,
                "is_anomaly": 1,
                "anomaly_type": "spike",
                "target_sensor": "temperature",
            }
        ]
    )

    load_dataframe_to_table(temp_connection, run_1_readings, "sensor_readings", run_id_1)
    load_dataframe_to_table(temp_connection, run_2_readings, "sensor_readings", run_id_2)

    response = client.get(f"/runs/{run_id_1}/machines/1/readings?limit=2")

    assert response.status_code == 200

    data = response.json()
    returned_steps = [reading["step"] for reading in data]

    assert len(data) == 2
    assert returned_steps == [1, 2]
    assert all(reading["run_id"] == run_id_1 for reading in data)
    assert all(reading["machine_id"] == 1 for reading in data)
    
def test_read_predictions_for_run(client, temp_connection):
    run_id_1 = insert_pipeline_run(temp_connection)
    run_id_2 = insert_pipeline_run(temp_connection)

    run_1_predictions = pd.DataFrame(
        [
            {
                "step": 1,
                "machine_id": 1,
                "real_value": 0,
                "prediction": 0,
                "anomaly_score": 0.10,
                "threshold": 0.35,
                "anomaly_type": "normal",
                "target_sensor": "temperature",
            },
            {
                "step": 2,
                "machine_id": 1,
                "real_value": 1,
                "prediction": 1,
                "anomaly_score": 0.90,
                "threshold": 0.35,
                "anomaly_type": "spike",
                "target_sensor": "temperature",
            },
            {
                "step": 3,
                "machine_id": 2,
                "real_value": 0,
                "prediction": 0,
                "anomaly_score": 0.20,
                "threshold": 0.35,
                "anomaly_type": "normal",
                "target_sensor": "pressure",
            },
        ]
    )

    run_2_predictions = pd.DataFrame(
        [
            {
                "step": 1,
                "machine_id": 1,
                "real_value": 1,
                "prediction": 1,
                "anomaly_score": 0.95,
                "threshold": 0.35,
                "anomaly_type": "drop",
                "target_sensor": "current",
            }
        ]
    )

    load_dataframe_to_table(temp_connection, run_1_predictions, "model_predictions", run_id_1)
    load_dataframe_to_table(temp_connection, run_2_predictions, "model_predictions", run_id_2)

    response = client.get(f"/runs/{run_id_1}/predictions?limit=2")

    assert response.status_code == 200

    data = response.json()
    returned_steps = [prediction["step"] for prediction in data]

    assert len(data) == 2
    assert returned_steps == [1, 2]
    assert all(prediction["run_id"] == run_id_1 for prediction in data)
    
def test_read_alert_events_returns_404_when_run_missing(client):
    response = client.get("/runs/999/events")

    assert response.status_code == 404
    assert response.json()["detail"] == "Pipeline run not found"
    
def test_read_predictions_returns_404_when_run_missing(client):
    response = client.get("/runs/999/predictions")

    assert response.status_code == 404
    assert response.json()["detail"] == "Pipeline run not found"
    
def test_read_sensor_readings_returns_404_when_run_missing(client):
    response = client.get("/runs/999/machines/1/readings")

    assert response.status_code == 404
    assert response.json()["detail"] == "Pipeline run not found"
    
def test_read_row_alerts_returns_404_when_run_missing(client):
    response = client.get("/runs/999/events/1/alerts")

    assert response.status_code == 404
    assert response.json()["detail"] == "Pipeline run not found"