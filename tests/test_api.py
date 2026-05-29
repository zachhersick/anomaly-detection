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