from pathlib import Path

import pandas as pd

from config import OUTPUT_DIR
from model import (
    build_chronological_splits,
    evaluate_final_test_set,
    find_cols_with_suffixes,
    load_feature_data,
    prepare_model_inputs,
    select_threshold,
    train_model,
)


INPUT_CSV = OUTPUT_DIR / "sensor_data_features.csv"
ABLATION_RESULTS_OUTPUT_PATH = OUTPUT_DIR / "ablation_results.csv"

ABLATION_GROUPS = {
    "lag_autocorr": ["_lag_5_autocorr", "_lag_10_autocorr"],
    "zero_cross": ["_centered_zero_cross_count_10", "_centered_zero_cross_count_20"],
    "center_balance": ["_center_balance_10", "_center_balance_20"],
    "trend_ratio": ["_trend_ratio_10", "_trend_ratio_25"],
}

ABLATION_RUNS = {
    "final_model": [],
    "without_lag_autocorr": ABLATION_GROUPS["lag_autocorr"],
    "without_zero_cross": ABLATION_GROUPS["zero_cross"],
    "without_center_balance": ABLATION_GROUPS["center_balance"],
    "without_trend_ratio": ABLATION_GROUPS["trend_ratio"],
}


def run_ablations(input_csv=INPUT_CSV, output_path=ABLATION_RESULTS_OUTPUT_PATH):
    df = load_feature_data(input_csv)
    X_full, y, meta = prepare_model_inputs(df)
    splits = build_chronological_splits(df)

    results = []
    for run_name, suffixes_to_drop in ABLATION_RUNS.items():
        cols_to_drop = find_cols_with_suffixes(X_full.columns, suffixes_to_drop)
        X_run = X_full.drop(columns=cols_to_drop)
        model = train_model(X_run.loc[splits.train_idx], y.loc[splits.train_idx])
        selection = select_threshold(
            model,
            X_run.loc[splits.validation_idx],
            y.loc[splits.validation_idx],
            meta.loc[splits.validation_idx],
        )
        validation_metrics = selection.validation_results.loc[
            selection.validation_results["threshold"] == selection.selected_threshold
        ].iloc[0]
        test_metrics, _, _ = evaluate_final_test_set(
            model,
            X_run.loc[splits.test_idx],
            y.loc[splits.test_idx],
            meta.loc[splits.test_idx],
            selection.selected_threshold,
        )
        results.append(
            {
                "run_name": run_name,
                "selected_threshold": selection.selected_threshold,
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_precision": validation_metrics["precision"],
                "validation_f1": validation_metrics["f1"],
                "validation_anomaly_recall": validation_metrics["anomaly_recall"],
                **{f"test_{key}": value for key, value in test_metrics.items() if key not in {"run_name", "threshold"}},
                "num_features_used": X_run.shape[1],
                "num_features_removed": len(cols_to_drop),
                "removed_features": ", ".join(cols_to_drop),
            }
        )

    ablation_df = pd.DataFrame(results)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ablation_df.to_csv(output_path, index=False)
    print("\nABLATION RESULTS")
    print(ablation_df)
    return ablation_df, splits


def main():
    run_ablations()


if __name__ == "__main__":
    main()
