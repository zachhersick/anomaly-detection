import pandas as pd

from evaluate import build_evaluation_tables, build_overall_metrics, build_recall_table, load_evaluation_inputs, run_evaluation


def make_predictions_df():
    return pd.DataFrame([
        {"step": 1, "machine_id": 1, "real_value": 1, "prediction": 1, "anomaly_type": "drift", "target_sensor": "temperature", "threshold": 0.35},
        {"step": 2, "machine_id": 1, "real_value": 1, "prediction": 0, "anomaly_type": "drift", "target_sensor": "temperature", "threshold": 0.35},
        {"step": 3, "machine_id": 2, "real_value": 1, "prediction": 1, "anomaly_type": "oscillation", "target_sensor": "current", "threshold": 0.35},
        {"step": 4, "machine_id": 2, "real_value": 0, "prediction": 1, "anomaly_type": "none", "target_sensor": "none", "threshold": 0.35},
        {"step": 5, "machine_id": 2, "real_value": 0, "prediction": 0, "anomaly_type": "none", "target_sensor": "none", "threshold": 0.35},
    ])


def test_load_evaluation_inputs_reads_predictions_only(tmp_path):
    predictions = make_predictions_df()
    path = tmp_path / "predictions.csv"
    predictions.to_csv(path, index=False)
    assert len(load_evaluation_inputs(path)) == len(predictions)


def test_build_recall_table_groups_and_calculates_recall():
    df = make_predictions_df()
    table = build_recall_table(df, df["real_value"] == 1, "anomaly_type")
    assert table.loc["drift", "recall"] == 0.5
    assert table.loc["oscillation", "recall"] == 1.0


def test_build_evaluation_tables_returns_expected_tables():
    tables = build_evaluation_tables(make_predictions_df())
    assert len(tables["false_positives"]) == 1
    assert len(tables["false_negatives"]) == 1
    assert tables["oscillation_by_sensor"].loc["current", "recall"] == 1.0


def test_build_overall_metrics_returns_expected_values():
    metrics = build_overall_metrics(make_predictions_df())
    assert metrics["accuracy"] == 0.6
    assert (metrics["true_negatives"], metrics["false_positives"], metrics["false_negatives"], metrics["true_positives"]) == (1, 1, 1, 2)


def test_run_evaluation_uses_final_predictions(tmp_path, capsys):
    path = tmp_path / "predictions.csv"
    make_predictions_df().to_csv(path, index=False)
    result = run_evaluation(path)
    assert "FINAL MODEL EVALUATION" in capsys.readouterr().out
    assert set(result) == {"predictions_df", "evaluation_tables", "overall_metrics"}
