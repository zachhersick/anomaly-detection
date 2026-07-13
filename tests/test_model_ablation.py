import pandas as pd

import model_ablation
from model import ThresholdSelection, build_chronological_splits


def make_feature_df():
    return pd.DataFrame(
        [
            {
                "step": step,
                "machine_id": machine_id,
                "any_anomaly": int(step % 4 == 0),
                "anomaly_type": "oscillation" if step % 4 == 0 else "none",
                "target_sensor": "current" if step % 4 == 0 else "none",
                "temperature": float(step),
                "current_lag_5_autocorr": 0.1,
            }
            for step in range(1, 301)
            for machine_id in range(1, 4)
        ]
    )


def test_ablations_share_boundaries_and_use_validation_thresholds(tmp_path, monkeypatch):
    input_path = tmp_path / "features.csv"
    output_path = tmp_path / "ablation_results.csv"
    df = make_feature_df()
    df.to_csv(input_path, index=False)
    expected = build_chronological_splits(df)
    validation_indices = []
    test_indices = []
    test_thresholds = []

    monkeypatch.setattr(model_ablation, "train_model", lambda X, y: object())

    def select(model, X, y, meta):
        validation_indices.append(tuple(X.index))
        return ThresholdSelection(
            0.55,
            pd.DataFrame([{"threshold": 0.55, "accuracy": 1.0, "precision": 1.0, "f1": 1.0, "anomaly_recall": 1.0}]),
        )

    def evaluate(model, X, y, meta, threshold):
        test_indices.append(tuple(X.index))
        test_thresholds.append(threshold)
        return ({"accuracy": 1.0, "precision": 1.0, "f1": 1.0, "anomaly_recall": 1.0, "true_negatives": 1, "false_positives": 0, "false_negatives": 0, "true_positives": 1, "oscillation_recall": 1.0, "current_osc_recall": 1.0, "voltage_osc_recall": 1.0}, pd.Series(dtype=int), pd.Series(dtype=float))

    monkeypatch.setattr(model_ablation, "select_threshold", select)
    monkeypatch.setattr(model_ablation, "evaluate_final_test_set", evaluate)
    results, _ = model_ablation.run_ablations(input_path, output_path)

    assert output_path.exists()
    assert all(indices == tuple(expected.validation_idx) for indices in validation_indices)
    assert all(indices == tuple(expected.test_idx) for indices in test_indices)
    assert test_thresholds == [0.55] * len(model_ablation.ABLATION_RUNS)
    assert set(results["selected_threshold"]) == {0.55}
