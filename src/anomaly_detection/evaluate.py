import pandas as pd
from sklearn import metrics


PREDICTIONS_INPUT_PATH = "outputs/predictions.csv"
DEBUG_OSCILLATION_DETAILS = False

OSC_FEATURE_SUFFIXES = [
    "lag_5_autocorr", "lag_10_autocorr", "centered_zero_cross_count_10",
    "centered_zero_cross_count_20", "dir_imbalance_10", "dir_imbalance_20",
    "trend_ratio_10", "trend_ratio_25", "center_balance_10", "center_balance_20",
]


def load_evaluation_inputs(predictions_path=PREDICTIONS_INPUT_PATH):
    """Load final untouched test predictions."""
    return pd.read_csv(predictions_path)


def build_recall_table(df, mask, group_col):
    rows = df[mask]
    summary = rows.groupby(by=group_col).agg(
        total=("prediction", "size"),
        correct=("prediction", lambda x: (x == 1).sum()),
        missed=("prediction", lambda x: (x == 0).sum()),
    )
    summary["recall"] = summary["correct"] / summary["total"]
    return summary


def build_evaluation_tables(predictions_df):
    anomaly_mask = predictions_df["real_value"] == 1
    drift_mask = anomaly_mask & (predictions_df["anomaly_type"] == "drift")
    oscillation_mask = anomaly_mask & (predictions_df["anomaly_type"] == "oscillation")
    false_positives = predictions_df[(predictions_df["real_value"] == 0) & (predictions_df["prediction"] == 1)]
    false_negatives = predictions_df[(predictions_df["real_value"] == 1) & (predictions_df["prediction"] == 0)]
    return {
        "anomaly_by_type": build_recall_table(predictions_df, anomaly_mask, "anomaly_type"),
        "anomaly_by_sensor": build_recall_table(predictions_df, anomaly_mask, "target_sensor"),
        "drift_by_sensor": build_recall_table(predictions_df, drift_mask, "target_sensor"),
        "oscillation_by_sensor": build_recall_table(predictions_df, oscillation_mask, "target_sensor"),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def build_overall_metrics(predictions_df):
    y_test = predictions_df["real_value"]
    predictions = predictions_df["prediction"]
    confusion_matrix = metrics.confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = confusion_matrix.ravel()
    return {
        "confusion_matrix": confusion_matrix,
        "accuracy": metrics.accuracy_score(y_test, predictions),
        "precision": metrics.precision_score(y_test, predictions, zero_division=0),
        "recall": metrics.recall_score(y_test, predictions, zero_division=0),
        "f1": metrics.f1_score(y_test, predictions, zero_division=0),
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "true_negatives": tn,
    }


def debug_oscillation_sensor(predictions_df, sensor):
    sensor_rows = predictions_df.loc[
        (predictions_df["real_value"] == 1)
        & (predictions_df["anomaly_type"] == "oscillation")
        & (predictions_df["target_sensor"] == sensor)
    ]
    compare_cols = [
        f"{sensor}_{suffix}"
        for suffix in OSC_FEATURE_SUFFIXES
        if f"{sensor}_{suffix}" in predictions_df.columns
    ]
    if not compare_cols:
        print(f"\n--- OSCILLATION DEBUG: {sensor} ---\nNo matching oscillation feature columns found.")
        return
    print(f"\n--- OSCILLATION DEBUG: {sensor} ---")
    print("correct:", (sensor_rows["prediction"] == 1).sum())
    print("missed:", (sensor_rows["prediction"] == 0).sum())
    print(sensor_rows.groupby("prediction")[compare_cols].mean())


def print_evaluation_report(predictions_df, evaluation_tables, overall_metrics, debug_oscillation_details=DEBUG_OSCILLATION_DETAILS):
    print("\n==============================\nFINAL MODEL EVALUATION\n==============================")
    if "threshold" in predictions_df.columns:
        print("threshold:", predictions_df["threshold"].iloc[0])
    print("\nConfusion Matrix\n", overall_metrics["confusion_matrix"])
    for name in ("accuracy", "precision", "recall", "f1", "false_positives", "false_negatives", "true_positives", "true_negatives"):
        print(f"{name.replace('_', ' ')}:", overall_metrics[name])
    for name in ("anomaly_by_type", "anomaly_by_sensor", "drift_by_sensor", "oscillation_by_sensor"):
        print(f"\n{name.replace('_', ' ').title()}")
        print(evaluation_tables[name].sort_values(by="recall"))
    print("\nFalse Positive Count:", len(evaluation_tables["false_positives"]))
    print("False Negative Count:", len(evaluation_tables["false_negatives"]))
    if debug_oscillation_details:
        debug_oscillation_sensor(predictions_df, "current")
        debug_oscillation_sensor(predictions_df, "voltage")


def run_evaluation(predictions_path=PREDICTIONS_INPUT_PATH, debug_oscillation_details=DEBUG_OSCILLATION_DETAILS):
    predictions_df = load_evaluation_inputs(predictions_path)
    evaluation_tables = build_evaluation_tables(predictions_df)
    overall_metrics = build_overall_metrics(predictions_df)
    print_evaluation_report(predictions_df, evaluation_tables, overall_metrics, debug_oscillation_details)
    return {"predictions_df": predictions_df, "evaluation_tables": evaluation_tables, "overall_metrics": overall_metrics}


def main():
    run_evaluation()


if __name__ == "__main__":
    main()
