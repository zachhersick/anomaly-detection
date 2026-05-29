from fastapi import Depends, FastAPI, HTTPException
import sqlite3

from db import get_connection
from db_queries import (
    get_latest_run_id,
    get_pipeline_runs,
    get_alert_events_for_run, 
    get_critical_alert_events,
)

app = FastAPI(
    title="Industrial Anomaly Detection API",
    description="API for reading pipeline runs, alert events, and anomaly detection ouputs.",
    version="0.1.0"
)

def get_db_connection():
    """
    create a database connection for one API request
    close it after the request finishes
    """
    conn = get_connection()
    
    try:
        yield conn
    finally:
        conn.close()

def row_to_dict(row: sqlite3.Row) -> dict:
    """
    convert one sqlite3.row into a normal dictionary
    """
    return dict(row)

def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    """
    convert a list of sqlitre3.row objects into a normal dictionary
    """
    return [row_to_dict(row) for row in rows]

@app.get("/health")
def health_check():
    """
    return basic api health status
    """
    return {"status": "ok"}

@app.get("/runs")
def read_pipeline_runs(conn=Depends(get_db_connection)):
    """
    return all stored pipeline runs, newest first
    """
    rows = get_pipeline_runs(conn)
    return rows_to_dicts(rows)

@app.get("/runs/latest")
def read_latest_run(conn=Depends(get_db_connection)):
    """
    return the latest pipeline run_id
    """
    latest_run_id = get_latest_run_id(conn)
    
    if latest_run_id is None:
        raise HTTPException(status_code=404, detail="No pipeline runs found.")
    
    return {"run_id": latest_run_id}

@app.get("/runs/{run_id}/events")
def read_alert_events_for_run(run_id, conn=Depends(get_db_connection)):
    """
    return grouped alert events for one pipeline run
    """
    rows = get_alert_events_for_run(conn, run_id)
    return rows_to_dicts(rows)
 
@app.get("/runs/{run_id}/events/critical")
def read_critical_alert_events_For_run(run_id, conn=Depends(get_db_connection)):
    """
    return critical grouped alert events for one pipeline run
    """
    rows = get_critical_alert_events(conn, run_id)
    return rows_to_dicts(rows)