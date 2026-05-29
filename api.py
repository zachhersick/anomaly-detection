from fastapi import Depends, FastAPI, HTTPException
import sqlite3

from db import get_connection
from db_queries import (
    get_latest_run_id,
    get_pipeline_runs,
    get_alert_events_for_run,
    get_critical_alert_events,
    get_row_alerts_for_event,
    get_sensor_readings_for_machine,
    get_predictions_for_run,
)


app = FastAPI(
    title="Industrial Anomaly Detection API",
    description="API for reading pipeline runs, alert events, and anomaly detection outputs.",
    version="0.1.0",
)


def get_db_connection():
    """
    Create a database connection for one API request.
    Close it after the request finishes.
    """
    conn = get_connection()

    try:
        yield conn
    finally:
        conn.close()


def row_to_dict(row: sqlite3.Row) -> dict:
    """
    Convert one sqlite3.Row into a normal dictionary.
    """
    return dict(row)


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    """
    Convert a list of sqlite3.Row objects into normal dictionaries.
    """
    return [row_to_dict(row) for row in rows]


@app.get("/health")
def health_check():
    """
    Return basic API health status.
    """
    return {"status": "ok"}


@app.get("/runs")
def read_pipeline_runs(conn=Depends(get_db_connection)):
    """
    Return all stored pipeline runs, newest first.
    """
    rows = get_pipeline_runs(conn)
    return rows_to_dicts(rows)


@app.get("/runs/latest")
def read_latest_run(conn=Depends(get_db_connection)):
    """
    Return the latest pipeline run_id.
    """
    latest_run_id = get_latest_run_id(conn)

    if latest_run_id is None:
        raise HTTPException(status_code=404, detail="No pipeline runs found.")

    return {"run_id": latest_run_id}


@app.get("/runs/{run_id}/events")
def read_alert_events_for_run(
    run_id: int,
    conn=Depends(get_db_connection),
):
    """
    Return grouped alert events for one pipeline run.
    """
    rows = get_alert_events_for_run(conn, run_id)
    return rows_to_dicts(rows)


@app.get("/runs/{run_id}/events/critical")
def read_critical_alert_events_for_run(
    run_id: int,
    conn=Depends(get_db_connection),
):
    """
    Return critical grouped alert events for one pipeline run.
    """
    rows = get_critical_alert_events(conn, run_id)
    return rows_to_dicts(rows)


@app.get("/runs/{run_id}/events/{event_id}/alerts")
def read_row_alerts_for_event(
    run_id: int,
    event_id: int,
    conn=Depends(get_db_connection),
):
    """
    Return individual row alerts inside one grouped alert event.
    """
    rows = get_row_alerts_for_event(conn, run_id, event_id)
    return rows_to_dicts(rows)


@app.get("/runs/{run_id}/machines/{machine_id}/readings")
def read_sensor_readings_for_machine(
    run_id: int,
    machine_id: int,
    limit: int | None = 100,
    conn=Depends(get_db_connection),
):
    """
    Return sensor readings for one machine in one run.
    """
    rows = get_sensor_readings_for_machine(
        conn=conn,
        run_id=run_id,
        machine_id=machine_id,
        limit=limit,
    )

    return rows_to_dicts(rows)


@app.get("/runs/{run_id}/predictions")
def read_predictions_for_run(
    run_id: int,
    limit: int | None = 100,
    conn=Depends(get_db_connection),
):
    """
    Return model predictions for one run.
    """
    rows = get_predictions_for_run(
        conn=conn,
        run_id=run_id,
        limit=limit,
    )

    return rows_to_dicts(rows)