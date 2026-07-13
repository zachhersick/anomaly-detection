import inspect
import json

import numpy as np
import pandas as pd
import pytest

from model import (
    THRESHOLDS,
    build_chronological_splits,
    evaluate_final_test_set,
    prepare_model_inputs,
    run_model_pipeline,
    select_threshold,
)


class FixedModel:
    def __init__(self, scores):
        self.scores = np.asarray(scores)

    def predict_proba(self, X):
        scores = self.scores[:len(X)]
        return np.column_stack([1 - scores, scores])


def make_feature_df(steps=300, machines=3):
    rows = []
    for step in range(1, steps + 1):
        for machine_id in range(1, machines + 1):
            anomaly = int(step % 4 == 0)
            rows.append(
                {
                    "step": step,
                    "machine_id": machine_id,
                    "any_anomaly": anomaly,
                    "anomaly_type": "oscillation" if anomaly else "none",
                    "target_sensor": "current" if anomaly else "none",
                    "temperature": float(step + machine_id),
                    "temperature_delta": float(anomaly),
                    "temperature_anomaly": anomaly,
                    "temperature_dir_imbalance_10": 0.1,
                }
            )
    return pd.DataFrame(rows)


def test_split_uses_step_not_row_position():
    df = make_feature_df().sample(frac=1, random_state=1)
    splits = build_chronological_splits(df)
    assert set(df.loc[splits.train_idx, "step"]) == set(splits.train_steps)


def test_splits_are_chronologically_ordered_and_disjoint():
    splits = build_chronological_splits(make_feature_df())
    assert max(splits.train_steps) < min(splits.validation_steps)
    assert max(splits.validation_steps) < min(splits.test_steps)
    assert set(splits.train_idx).isdisjoint(splits.validation_idx)
    assert set(splits.train_idx).isdisjoint(splits.test_idx)
    assert set(splits.validation_idx).isdisjoint(splits.test_idx)


def test_purge_gaps_are_excluded_and_exactly_fifty_steps():
    df = make_feature_df()
    splits = build_chronological_splits(df)
    assigned_steps = set(splits.train_steps + splits.validation_steps + splits.test_steps)
    assert len(splits.first_purge_steps) == 50
    assert len(splits.second_purge_steps) == 50
    assert not (set(splits.first_purge_steps) | set(splits.second_purge_steps)) & assigned_steps


def test_every_machine_uses_the_same_boundaries():
    df = make_feature_df()
    splits = build_chronological_splits(df)
    for machine_id in df["machine_id"].unique():
        machine_rows = df[df["machine_id"] == machine_id]
        assert set(machine_rows.loc[splits.train_idx.intersection(machine_rows.index), "step"]) == set(splits.train_steps)
        assert set(machine_rows.loc[splits.validation_idx.intersection(machine_rows.index), "step"]) == set(splits.validation_steps)
        assert set(machine_rows.loc[splits.test_idx.intersection(machine_rows.index), "step"]) == set(splits.test_steps)


def test_shuffling_rows_does_not_change_membership():
    df = make_feature_df()
    ordered = build_chronological_splits(df)
    shuffled = build_chronological_splits(df.sample(frac=1, random_state=2))
    assert set(ordered.train_idx) == set(shuffled.train_idx)
    assert set(ordered.validation_idx) == set(shuffled.validation_idx)
    assert set(ordered.test_idx) == set(shuffled.test_idx)
    assert ordered.train_steps == shuffled.train_steps


def test_all_rows_are_assigned_to_a_split_or_purge_gap():
    df = make_feature_df()
    splits = build_chronological_splits(df)
    purge_idx = df.index[df["step"].isin(splits.first_purge_steps + splits.second_purge_steps)]
    assigned = set(splits.train_idx) | set(splits.validation_idx) | set(splits.test_idx) | set(purge_idx)
    assert assigned == set(df.index)


@pytest.mark.parametrize(
    "df, kwargs",
    [
        (pd.DataFrame({"other": [1]}), {}),
        (pd.DataFrame({"step": []}), {}),
        (make_feature_df(), {"train_fraction": 0.6, "validation_fraction": 0.2, "test_fraction": 0.1}),
        (make_feature_df(), {"train_fraction": 0}),
        (make_feature_df(), {"validation_fraction": -0.1}),
        (make_feature_df(), {"purge_gap_steps": -1}),
        (make_feature_df(steps=102), {}),
    ],
)
def test_split_rejects_invalid_inputs(df, kwargs):
    with pytest.raises(ValueError):
        build_chronological_splits(df, **kwargs)


def test_threshold_selection_uses_highest_validation_f1():
    X = pd.DataFrame({"feature": range(4)})
    y = pd.Series([1, 1, 0, 0])
    meta = pd.DataFrame({"anomaly_type": ["none"] * 4, "target_sensor": ["none"] * 4})
    selection = select_threshold(FixedModel([0.9, 0.5, 0.4, 0.2]), X, y, meta)
    assert selection.selected_threshold == 0.45


def test_threshold_selection_tie_breaking_is_deterministic():
    X = pd.DataFrame({"feature": range(4)})
    y = pd.Series([1, 1, 0, 0])
    meta = pd.DataFrame({"anomaly_type": ["none"] * 4, "target_sensor": ["none"] * 4})
    selection = select_threshold(FixedModel([0.9, 0.4, 0.4, 0.4]), X, y, meta, [0.35, 0.45])
    assert selection.selected_threshold == 0.35


def test_threshold_selection_accepts_no_test_data():
    parameters = inspect.signature(select_threshold).parameters
    assert not {"X_test", "y_test", "meta_test"} & set(parameters)


def test_changing_test_data_cannot_change_selected_threshold():
    X = pd.DataFrame({"feature": range(4)})
    y = pd.Series([1, 1, 0, 0])
    meta = pd.DataFrame({"anomaly_type": ["none"] * 4, "target_sensor": ["none"] * 4})
    first = select_threshold(FixedModel([0.9, 0.5, 0.4, 0.2]), X, y, meta)
    second = select_threshold(FixedModel([0.9, 0.5, 0.4, 0.2]), X, y, meta)
    assert first.selected_threshold == second.selected_threshold


def test_final_test_predictions_use_selected_threshold_and_metrics_match():
    X = pd.DataFrame({"feature": range(4)})
    y = pd.Series([1, 0, 1, 0])
    meta = pd.DataFrame({"anomaly_type": ["oscillation", "none", "oscillation", "none"], "target_sensor": ["current", "none", "voltage", "none"]})
    result, predictions, scores = evaluate_final_test_set(FixedModel([0.8, 0.7, 0.2, 0.1]), X, y, meta, 0.5)
    assert predictions.tolist() == [int(score >= 0.5) for score in scores]
    assert (result["true_negatives"], result["false_positives"], result["false_negatives"], result["true_positives"]) == (1, 1, 1, 1)
    assert result["accuracy"] == 0.5
    assert result["precision"] == 0.5
    assert result["anomaly_recall"] == 0.5
    assert result["f1"] == 0.5


def test_prepare_model_inputs_excludes_labels_and_metadata():
    X, y, meta = prepare_model_inputs(make_feature_df())
    assert "any_anomaly" not in X
    assert "temperature_anomaly" not in X
    assert "temperature_dir_imbalance_10" not in X
    assert len(X) == len(y) == len(meta)


def test_pipeline_writes_final_test_outputs_and_metadata(tmp_path):
    input_path = tmp_path / "sensor_data_features.csv"
    predictions_path = tmp_path / "predictions.csv"
    validation_path = tmp_path / "validation_threshold_results.csv"
    test_metrics_path = tmp_path / "test_metrics.csv"
    importance_path = tmp_path / "feature_importance.csv"
    make_feature_df().to_csv(input_path, index=False)
    _, validation_results, predictions, _ = run_model_pipeline(
        input_csv=input_path,
        predictions_output_path=predictions_path,
        validation_threshold_results_output_path=validation_path,
        test_metrics_output_path=test_metrics_path,
        feature_importance_output_path=importance_path,
    )
    test_metrics = pd.read_csv(test_metrics_path)
    with open("artifacts/model_metadata.json") as file:
        metadata = json.load(file)
    assert len(validation_results) == len(THRESHOLDS)
    assert len(test_metrics) == 1
    assert len(predictions) == metadata["test_rows"]
    assert set(predictions["threshold"]) == {metadata["threshold"]}
    assert metadata["first_purge_step_end"] - metadata["first_purge_step_start"] + 1 == 50
    assert metadata["second_purge_step_end"] - metadata["second_purge_step_start"] + 1 == 50
