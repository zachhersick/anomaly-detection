import pandas as pd
import pytest

from alert_events import group_alert_events, output_cols, MAX_STEP_GAP


def make_alert(
    alert_id=1,
    step=10,
    machine_id=1,
    sensor="temperature",
    sensor_value=105.0,
    prediction=1,
    anomaly_score=0.81,
    severity="WARNING",
    alert_type="model_anomaly",
    reason="model flagged anomaly",
    status="open",
    anomaly_type="spike",
    real_value=1,
):
    return {
        "alert_id": alert_id,
        "step": step,
        "machine_id": machine_id,
        "sensor": sensor,
        "sensor_value": sensor_value,
        "prediction": prediction,
        "anomaly_score": anomaly_score,
        "severity": severity,
        "alert_type": alert_type,
        "reason": reason,
        "status": status,
        "anomaly_type": anomaly_type,
        "real_value": real_value,
    }


def test_one_alert_creates_one_event():
    alerts = pd.DataFrame([
        make_alert()
    ])

    events_df = group_alert_events(alerts)

    assert len(events_df) == 1

    event = events_df.iloc[0]

    assert event["event_id"] == 1
    assert event["machine_id"] == 1
    assert event["sensor"] == "temperature"
    assert event["anomaly_type"] == "spike"
    assert event["start_step"] == 10
    assert event["end_step"] == 10
    assert event["duration"] == 1
    assert event["alert_count"] == 1
    assert event["max_severity"] == "WARNING"
    assert event["first_reason"] == "model flagged anomaly"
    assert event["max_anomaly_score"] == 0.81
    assert event["mean_anomaly_score"] == 0.81
    assert event["min_sensor_value"] == 105.0
    assert event["max_sensor_value"] == 105.0


def test_consecutive_alerts_group_into_one_event():
    alerts = pd.DataFrame([
        make_alert(
            alert_id=1,
            step=10,
            sensor_value=105.0,
            anomaly_score=0.81,
            severity="WARNING",
            reason="model flagged anomaly",
        ),
        make_alert(
            alert_id=2,
            step=11,
            sensor_value=108.0,
            anomaly_score=0.90,
            severity="CRITICAL",
            reason="critical temperature threshold exceeded",
        ),
    ])

    events_df = group_alert_events(alerts)

    assert len(events_df) == 1

    event = events_df.iloc[0]

    assert event["start_step"] == 10
    assert event["end_step"] == 11
    assert event["duration"] == 2
    assert event["alert_count"] == 2
    assert event["max_anomaly_score"] == 0.90
    assert event["mean_anomaly_score"] == pytest.approx(0.855)
    assert event["max_severity"] == "CRITICAL"
    assert event["max_severity_reason"] == "critical temperature threshold exceeded"
    assert event["min_sensor_value"] == 105.0
    assert event["max_sensor_value"] == 108.0


def test_gap_larger_than_max_step_gap_creates_new_event():
    first_step = 10
    second_step = first_step + MAX_STEP_GAP + 1

    alerts = pd.DataFrame([
        make_alert(alert_id=1, step=first_step),
        make_alert(alert_id=2, step=second_step, anomaly_score=0.90),
    ])

    events_df = group_alert_events(alerts)

    assert len(events_df) == 2

    event_1 = events_df.iloc[0]
    event_2 = events_df.iloc[1]

    assert event_1["start_step"] == first_step
    assert event_1["end_step"] == first_step
    assert event_1["alert_count"] == 1

    assert event_2["start_step"] == second_step
    assert event_2["end_step"] == second_step
    assert event_2["alert_count"] == 1


def test_different_anomaly_type_creates_new_event():
    alerts = pd.DataFrame([
        make_alert(alert_id=1, step=10, anomaly_type="spike"),
        make_alert(alert_id=2, step=11, anomaly_type="drop"),
    ])

    events_df = group_alert_events(alerts)

    assert len(events_df) == 2
    assert set(events_df["anomaly_type"]) == {"spike", "drop"}
    assert all(events_df["alert_count"] == 1)


def test_different_machine_id_creates_new_event():
    alerts = pd.DataFrame([
        make_alert(alert_id=1, step=10, machine_id=1),
        make_alert(alert_id=2, step=11, machine_id=2),
    ])

    events_df = group_alert_events(alerts)

    assert len(events_df) == 2
    assert set(events_df["machine_id"]) == {1, 2}
    assert all(events_df["alert_count"] == 1)


def test_different_sensor_creates_new_event():
    alerts = pd.DataFrame([
        make_alert(alert_id=1, step=10, sensor="temperature"),
        make_alert(alert_id=2, step=11, sensor="pressure"),
    ])

    events_df = group_alert_events(alerts)

    assert len(events_df) == 2
    assert set(events_df["sensor"]) == {"temperature", "pressure"}
    assert all(events_df["alert_count"] == 1)


def test_highest_severity_reason_is_preserved():
    alerts = pd.DataFrame([
        make_alert(
            alert_id=1,
            step=10,
            severity="INFO",
            reason="model-only low severity alert",
        ),
        make_alert(
            alert_id=2,
            step=11,
            severity="WARNING",
            reason="warning threshold exceeded",
        ),
        make_alert(
            alert_id=3,
            step=12,
            severity="CRITICAL",
            reason="critical threshold exceeded",
        ),
    ])

    events_df = group_alert_events(alerts)

    assert len(events_df) == 1

    event = events_df.iloc[0]

    assert event["max_severity"] == "CRITICAL"
    assert event["max_severity_reason"] == "critical threshold exceeded"
    assert event["first_reason"] == "model-only low severity alert"


def test_empty_alerts_returns_empty_events_with_expected_columns():
    alerts = pd.DataFrame(columns=[
        "alert_id",
        "step",
        "machine_id",
        "sensor",
        "sensor_value",
        "prediction",
        "anomaly_score",
        "severity",
        "alert_type",
        "reason",
        "status",
        "anomaly_type",
        "real_value",
    ])

    events_df = group_alert_events(alerts)

    assert len(events_df) == 0
    assert list(events_df.columns) == output_cols
