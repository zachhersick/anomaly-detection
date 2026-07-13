from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from config import (
    ARTIFACT_DIR,
    OUTPUT_DIR,
    PURGE_GAP_STEPS,
    RANDOM_STATE,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
)


INPUT_CSV = OUTPUT_DIR / "sensor_data_features.csv"
PREDICTIONS_OUTPUT_PATH = OUTPUT_DIR / "predictions.csv"
FEATURE_IMPORTANCE_OUTPUT_PATH = OUTPUT_DIR / "feature_importance.csv"
VALIDATION_THRESHOLD_RESULTS_OUTPUT_PATH = OUTPUT_DIR / "validation_threshold_results.csv"
TEST_METRICS_OUTPUT_PATH = OUTPUT_DIR / "test_metrics.csv"

LABEL_COL = "any_anomaly"

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

PERMANENT_DROP_SUFFIXES = ["_dir_imbalance_10", "_dir_imbalance_20"]


@dataclass(frozen=True)
class ChronologicalSplits:
    train_idx: pd.Index
    validation_idx: pd.Index
    test_idx: pd.Index
    train_steps: tuple[int, ...]
    first_purge_steps: tuple[int, ...]
    validation_steps: tuple[int, ...]
    second_purge_steps: tuple[int, ...]
    test_steps: tuple[int, ...]


@dataclass(frozen=True)
class ThresholdSelection:
    selected_threshold: float
    validation_results: pd.DataFrame


def find_cols_with_suffixes(columns, suffixes):
    return [
        column
        for column in columns
        if any(column.endswith(suffix) for suffix in suffixes)
    ]


def make_model():
    return RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )


def safe_recall(correct, total):
    return np.nan if total == 0 else correct / total


def evaluate_predictions(run_name, y_true, pred_series, meta, threshold=None):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, pred_series, labels=[0, 1]).ravel()
    osc_mask = (y_true == 1) & (meta["anomaly_type"] == "oscillation")
    current_osc_mask = osc_mask & (meta["target_sensor"] == "current")
    voltage_osc_mask = osc_mask & (meta["target_sensor"] == "voltage")

    def recall(mask):
        total = mask.sum()
        return safe_recall(((pred_series == 1) & mask).sum(), total)

    return {
        "run_name": run_name,
        "threshold": threshold,
        "accuracy": metrics.accuracy_score(y_true, pred_series),
        "precision": metrics.precision_score(y_true, pred_series, zero_division=0),
        "f1": metrics.f1_score(y_true, pred_series, zero_division=0),
        "anomaly_recall": safe_recall(tp, tp + fn),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "oscillation_recall": recall(osc_mask),
        "current_osc_recall": recall(current_osc_mask),
        "voltage_osc_recall": recall(voltage_osc_mask),
    }


def build_chronological_splits(
    df: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
    test_fraction: float = TEST_FRACTION,
    purge_gap_steps: int = PURGE_GAP_STEPS,
) -> ChronologicalSplits:
    if df.empty:
        raise ValueError("Cannot split an empty dataframe.")
    if "step" not in df.columns:
        raise ValueError("Missing step column.")
    if any(fraction <= 0 for fraction in (train_fraction, validation_fraction, test_fraction)):
        raise ValueError("Split fractions must be positive.")
    if not math.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
        raise ValueError("Split fractions must total 1.0.")
    if purge_gap_steps < 0:
        raise ValueError("Purge gap must be non-negative.")

    unique_steps = sorted(df["step"].unique())
    usable_step_count = len(unique_steps) - (2 * purge_gap_steps)
    train_step_count = int(usable_step_count * train_fraction)
    validation_step_count = int(usable_step_count * validation_fraction)
    test_step_count = usable_step_count - train_step_count - validation_step_count
    if min(train_step_count, validation_step_count, test_step_count) <= 0:
        raise ValueError("Insufficient timesteps for three splits and purge gaps.")

    train_end = train_step_count
    first_purge_end = train_end + purge_gap_steps
    validation_end = first_purge_end + validation_step_count
    second_purge_end = validation_end + purge_gap_steps
    train_steps = tuple(unique_steps[:train_end])
    first_purge_steps = tuple(unique_steps[train_end:first_purge_end])
    validation_steps = tuple(unique_steps[first_purge_end:validation_end])
    second_purge_steps = tuple(unique_steps[validation_end:second_purge_end])
    test_steps = tuple(unique_steps[second_purge_end:])

    return ChronologicalSplits(
        train_idx=df.index[df["step"].isin(train_steps)],
        validation_idx=df.index[df["step"].isin(validation_steps)],
        test_idx=df.index[df["step"].isin(test_steps)],
        train_steps=train_steps,
        first_purge_steps=first_purge_steps,
        validation_steps=validation_steps,
        second_purge_steps=second_purge_steps,
        test_steps=test_steps,
    )


def load_feature_data(input_csv=INPUT_CSV):
    df = pd.read_csv(input_csv).replace([np.inf, -np.inf], np.nan).dropna().copy()
    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing label column: {LABEL_COL}")
    return df


def prepare_model_inputs(df):
    metadata_cols = ["step", "timestamp", "machine_id", "anomaly_type", "target_sensor"]
    non_feature_cols = metadata_cols + [LABEL_COL] + [
        column for column in df.columns if column.endswith("_anomaly")
    ]
    non_feature_cols = [column for column in non_feature_cols if column in df.columns]
    drop_cols = non_feature_cols + find_cols_with_suffixes(df.columns, PERMANENT_DROP_SUFFIXES)
    return (
        df.drop(columns=drop_cols).select_dtypes(include=[np.number]),
        df[LABEL_COL],
        df[[column for column in metadata_cols if column in df.columns]].copy(),
    )


def train_model(X_train, y_train):
    model = make_model()
    model.fit(X_train, y_train)
    return model


def select_threshold(
    model,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    meta_validation: pd.DataFrame,
    thresholds: list[float] = THRESHOLDS,
) -> ThresholdSelection:
    validation_scores = pd.Series(model.predict_proba(X_validation)[:, 1], index=X_validation.index)
    results = []
    for threshold in thresholds:
        predictions = (validation_scores >= threshold).astype(int)
        results.append(
            evaluate_predictions(
                f"threshold_{threshold}", y_validation, predictions, meta_validation, threshold
            )
        )
    validation_results = pd.DataFrame(results)
    selected_threshold = validation_results.sort_values(
        by=["f1", "anomaly_recall", "false_positives", "threshold"],
        ascending=[False, False, True, True],
    ).iloc[0]["threshold"]
    return ThresholdSelection(float(selected_threshold), validation_results)


def evaluate_final_test_set(model, X_test, y_test, meta_test, selected_threshold):
    test_scores = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index, name="anomaly_score")
    test_predictions = (test_scores >= selected_threshold).astype(int).rename("prediction")
    metrics_result = evaluate_predictions(
        "final_test", y_test, test_predictions, meta_test, selected_threshold
    )
    return metrics_result, test_predictions, test_scores


def save_dataframe(df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_feature_importance(model, feature_columns, output_path=FEATURE_IMPORTANCE_OUTPUT_PATH):
    if not hasattr(model, "feature_importances_"):
        return None
    feature_importance = pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_})
    feature_importance = feature_importance.sort_values(by="importance", ascending=False)
    save_dataframe(feature_importance, output_path)
    return feature_importance


def step_range(steps):
    return (int(steps[0]), int(steps[-1])) if steps else (None, None)


def run_model_pipeline(
    input_csv=INPUT_CSV,
    predictions_output_path=PREDICTIONS_OUTPUT_PATH,
    validation_threshold_results_output_path=VALIDATION_THRESHOLD_RESULTS_OUTPUT_PATH,
    test_metrics_output_path=TEST_METRICS_OUTPUT_PATH,
    feature_importance_output_path=FEATURE_IMPORTANCE_OUTPUT_PATH,
):
    df = load_feature_data(input_csv)
    X, y, meta = prepare_model_inputs(df)
    splits = build_chronological_splits(df)
    X_train, y_train = X.loc[splits.train_idx], y.loc[splits.train_idx]
    X_validation, y_validation = X.loc[splits.validation_idx], y.loc[splits.validation_idx]
    X_test, y_test = X.loc[splits.test_idx], y.loc[splits.test_idx]
    meta_validation, meta_test = meta.loc[splits.validation_idx], meta.loc[splits.test_idx]

    model = train_model(X_train, y_train)
    threshold_selection = select_threshold(model, X_validation, y_validation, meta_validation)
    test_metrics, test_predictions, test_scores = evaluate_final_test_set(
        model, X_test, y_test, meta_test, threshold_selection.selected_threshold
    )
    predictions_df = df.loc[splits.test_idx].copy()
    predictions_df["real_value"] = y_test
    predictions_df["prediction"] = test_predictions
    predictions_df["anomaly_score"] = test_scores
    predictions_df["threshold"] = threshold_selection.selected_threshold

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "model.joblib")
    feature_columns = list(X_train.columns)
    with open(ARTIFACT_DIR / "feature_columns.json", "w") as file:
        json.dump(feature_columns, file, indent=4)
    metadata = {
        "model_type": "RandomForestClassifier",
        "threshold": threshold_selection.selected_threshold,
        "threshold_selection_metric": "validation_f1",
        "random_state": RANDOM_STATE,
        "train_fraction": TRAIN_FRACTION,
        "validation_fraction": VALIDATION_FRACTION,
        "test_fraction": TEST_FRACTION,
        "purge_gap_steps": PURGE_GAP_STEPS,
        "feature_count": len(feature_columns),
        "training_rows": len(X_train),
        "validation_rows": len(X_validation),
        "test_rows": len(X_test),
        "train_step_start": step_range(splits.train_steps)[0],
        "train_step_end": step_range(splits.train_steps)[1],
        "first_purge_step_start": step_range(splits.first_purge_steps)[0],
        "first_purge_step_end": step_range(splits.first_purge_steps)[1],
        "validation_step_start": step_range(splits.validation_steps)[0],
        "validation_step_end": step_range(splits.validation_steps)[1],
        "second_purge_step_start": step_range(splits.second_purge_steps)[0],
        "second_purge_step_end": step_range(splits.second_purge_steps)[1],
        "test_step_start": step_range(splits.test_steps)[0],
        "test_step_end": step_range(splits.test_steps)[1],
        "created_at": datetime.now().isoformat(),
    }
    with open(ARTIFACT_DIR / "model_metadata.json", "w") as file:
        json.dump(metadata, file, indent=4)

    save_dataframe(predictions_df, predictions_output_path)
    save_dataframe(threshold_selection.validation_results, validation_threshold_results_output_path)
    save_dataframe(pd.DataFrame([test_metrics]), test_metrics_output_path)
    feature_importance_df = save_feature_importance(model, feature_columns, feature_importance_output_path)

    print("\nVALIDATION THRESHOLD RESULTS")
    print(threshold_selection.validation_results)
    print("\nFINAL TEST RESULTS")
    print(pd.DataFrame([test_metrics]))
    print("Selected threshold:", threshold_selection.selected_threshold)
    for name, steps in (
        ("Training", splits.train_steps),
        ("First purge", splits.first_purge_steps),
        ("Validation", splits.validation_steps),
        ("Second purge", splits.second_purge_steps),
        ("Test", splits.test_steps),
    ):
        print(f"{name} step range: {step_range(steps)}")
    print("Training row count:", len(X_train))
    print("Validation row count:", len(X_validation))
    print("Test row count:", len(X_test))
    return model, threshold_selection.validation_results, predictions_df, feature_importance_df


def main():
    run_model_pipeline()


if __name__ == "__main__":
    main()
