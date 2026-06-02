import sqlite3

def get_latest_run_id(conn):
    """
    Return the most recent pipeline run_id
    """
    row = conn.execute(
        """
        SELECT run_id
        FROM pipeline_runs
        ORDER BY run_id DESC
        LIMIT 1
        """
    ).fetchone()
    
    if row is None:
        return None
    
    return row['run_id']

def get_pipeline_runs(conn):
    """
    return all pipeline runs
    """
    rows = conn.execute(
        """
        SELECT *
        FROM pipeline_runs
        ORDER BY run_id DESC
        """
    ).fetchall()
    
    return rows

def get_alert_events_for_run(conn, run_id):
    """
    return grouped alert events for one pipeline run
    """
    rows = conn.execute(
        """
        SELECT *
        FROM alert_events
        WHERE run_id = ?
        ORDER BY start_step ASC
        """
    , (run_id, )).fetchall()
    
    return rows

def get_critical_alert_events(conn, run_id):
    """
    return only critical alert events for one run
    """
    rows = conn.execute(
        """
        SELECT *
        FROM alert_events
        WHERE run_id = ?
            AND max_severity = ?
        ORDER BY start_step ASC
        """
    , (run_id, 'CRITICAL', )).fetchall()
    
    return rows

def get_row_alerts_for_event(conn, run_id, event_id):
    """
    given one grouped event, return the individual row alerts inside that event
    """
    event_row = conn.execute(
        """
        SELECT *
        FROM alert_events
        WHERE run_id = ?
            AND event_id = ?
        """
    , (run_id, event_id, )).fetchone()
    
    if event_row is None:
        return []
    
    event_machine_id = event_row['machine_id']
    event_sensor = event_row['sensor']
    event_anomaly_type = event_row['anomaly_type']
    event_start_step = event_row['start_step']
    event_end_step = event_row['end_step']
    
    alert_rows = conn.execute(
        """
        SELECT *
        FROM row_alerts
        WHERE run_id = ?
            AND machine_id = ?
            AND sensor = ?
            AND anomaly_type = ?
            AND step BETWEEN ? AND ?
        ORDER BY step ASC
        """
    , (run_id, event_machine_id, event_sensor, event_anomaly_type, event_start_step, event_end_step, )).fetchall()
    
    return alert_rows

def get_sensor_readings_for_machine(conn, run_id, machine_id, limit=None):
    """
    return sensor readings for one machine in one run
    """
    if limit is None:
        rows = conn.execute(
            """
            SELECT *
            FROM sensor_readings
            WHERE run_id = ?
                AND machine_id = ?
            ORDER BY step ASC
            """
        , (run_id, machine_id, )).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT *
            FROM sensor_readings
            WHERE run_id = ?
                AND machine_id = ?
            ORDER BY step ASC
            LIMIT ?
            """
        , (run_id, machine_id, limit, )).fetchall()
    
    return rows

def get_predictions_for_run(conn, run_id, limit=None):
    """
    return model predictions for one run
    """
    if limit is None:
        rows = conn.execute(
            """
            SELECT *
            FROM model_predictions
            WHERE run_id = ?
            ORDER BY step ASC, machine_id ASC, target_sensor ASC
            """
        , (run_id, )).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT *
            FROM model_predictions
            WHERE run_id = ?
            ORDER BY step ASC, machine_id ASC, target_sensor ASC
            LIMIT ?
            """
        , (run_id, limit, )).fetchall()
    
    return rows

def run_exists(conn, run_id):
    """
    Return True if a pipeline run exists.
    Return False otherwise.
    """
    row = conn.execute(
        """
        SELECT 1
        FROM pipeline_runs
        WHERE run_id = ?
        LIMIT 1
        """,
        (run_id,),
    ).fetchone()

    return row is not None

def event_exists(conn,run_id, event_id):
    """
    Return True if an alert event exists.
    Return False otherwise.
    """
    row = conn.execute(
        """
        SELECT 1
        FROM alert_events
        WHERE run_id = ?
            AND event_id = ?
        LIMIT 1
        """,
        (run_id, event_id, ),
    ).fetchone()

    return row is not None

def get_filtered_alert_events_for_run(conn, run_id, severity=None, sensor=None, anomaly_type=None, limit=100):
    sql = """
        SELECT *
        FROM alert_events
        WHERE run_id = ?
    """
    
    params = [run_id]
    
    if severity is not None:
        sql += " AND max_severity = ?"
        params.append(severity)
    
    if sensor is not None:
        sql += " AND sensor = ?"
        params.append(sensor)
    
    if anomaly_type is not None:
        sql += " AND anomaly_type = ?"
        params.append(anomaly_type)
        
    sql += " ORDER BY start_step ASC"
    
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    
    rows = conn.execute(sql, params).fetchall()
    return rows