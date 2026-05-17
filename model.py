import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

INPUT_CSV = 'sensor_data_features.csv'

LABEL_COL = 'any_anomaly'

RANDOM_STATE = 42
TEST_SIZE = 0.2

THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]

DEFAULT_THRESHOLD = 0.35


# ---------------------------------------------------------------------
# Ablation groups
# ---------------------------------------------------------------------

ABLATION_GROUPS = {
    'lag_autocorr': [
        '_lag_5_autocorr',
        '_lag_10_autocorr',
    ],
    'zero_cross': [
        '_centered_zero_cross_count_10',
        '_centered_zero_cross_count_20',
    ],
    'center_balance': [
        '_center_balance_10',
        '_center_balance_20',
    ],
    'dir_imbalance': [
        '_dir_imbalance_10',
        '_dir_imbalance_20',
    ],
    'trend_ratio': [
        '_trend_ratio_10',
        '_trend_ratio_25',
    ],
}

PERMANENT_DROP_SUFFIXES = [
    '_dir_imbalance_10',
    '_dir_imbalance_20',
]

ABLATION_RUNS = {
    'final_model': [],
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def find_cols_with_suffixes(columns, suffixes):
    cols_to_drop = []

    for col in columns:
        for suffix in suffixes:
            if col.endswith(suffix):
                cols_to_drop.append(col)
                break

    return cols_to_drop


def make_model():
    return RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1
    )


def safe_recall(correct, total):
    if total == 0:
        return np.nan
    return correct / total


def evaluate_predictions(run_name, y_test, pred_series, meta_test, threshold=None):
    cm = metrics.confusion_matrix(y_test, pred_series, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    anomaly_recall = safe_recall(tp, tp + fn)

    precision = metrics.precision_score(
        y_test,
        pred_series,
        zero_division=0
    )

    f1 = metrics.f1_score(
        y_test,
        pred_series,
        zero_division=0
    )

    osc_mask = (
        (y_test == 1) &
        (meta_test['anomaly_type'] == 'oscillation')
    )

    current_osc_mask = (
        osc_mask &
        (meta_test['target_sensor'] == 'current')
    )

    voltage_osc_mask = (
        osc_mask &
        (meta_test['target_sensor'] == 'voltage')
    )

    osc_total = osc_mask.sum()
    osc_correct = ((pred_series == 1) & osc_mask).sum()

    current_osc_total = current_osc_mask.sum()
    current_osc_correct = ((pred_series == 1) & current_osc_mask).sum()

    voltage_osc_total = voltage_osc_mask.sum()
    voltage_osc_correct = ((pred_series == 1) & voltage_osc_mask).sum()

    return {
        'run_name': run_name,
        'threshold': threshold,

        'accuracy': metrics.accuracy_score(y_test, pred_series),
        'precision': precision,
        'f1': f1,

        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,

        'anomaly_recall': anomaly_recall,

        'oscillation_correct': osc_correct,
        'oscillation_total': osc_total,
        'oscillation_recall': safe_recall(osc_correct, osc_total),

        'current_osc_correct': current_osc_correct,
        'current_osc_total': current_osc_total,
        'current_osc_recall': safe_recall(current_osc_correct, current_osc_total),

        'voltage_osc_correct': voltage_osc_correct,
        'voltage_osc_total': voltage_osc_total,
        'voltage_osc_recall': safe_recall(voltage_osc_correct, voltage_osc_total),
    }


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

df = pd.read_csv(INPUT_CSV)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna().copy()

if LABEL_COL not in df.columns:
    raise ValueError(f'Missing label column: {LABEL_COL}')


# ---------------------------------------------------------------------
# Build X, y, and metadata
# ---------------------------------------------------------------------

metadata_cols = [
    'step',
    'timestamp',
    'machine_id',
    'anomaly_type',
    'target_sensor',
]

non_feature_cols = [
    'step',
    'timestamp',
    'machine_id',
    LABEL_COL,
    'anomaly_type',
    'target_sensor',
]

# Remove per-sensor label columns from X.
for col in df.columns:
    if col.endswith('_anomaly'):
        non_feature_cols.append(col)

non_feature_cols = [
    col for col in non_feature_cols
    if col in df.columns
]

metadata_cols = [
    col for col in metadata_cols
    if col in df.columns
]

y = df[LABEL_COL]

permanent_drop_cols = find_cols_with_suffixes(
    df.columns,
    PERMANENT_DROP_SUFFIXES
)

drop_cols = non_feature_cols + permanent_drop_cols

X_full = df.drop(columns=drop_cols)

# Keep only numeric features.
X_full = X_full.select_dtypes(include=[np.number])

meta = df[metadata_cols].copy()


# ---------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------

train_idx, test_idx = train_test_split(
    df.index,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

y_train = y.loc[train_idx]
y_test = y.loc[test_idx]

meta_test = meta.loc[test_idx].copy()


# ---------------------------------------------------------------------
# Train final model
# ---------------------------------------------------------------------

model = make_model()
model.fit(X_full.loc[train_idx], y_train)

X_test = X_full.loc[test_idx]

anomaly_scores = model.predict_proba(X_test)[:, 1]

score_series = pd.Series(
    anomaly_scores,
    index=test_idx,
    name='anomaly_score'
)


# ---------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------

threshold_results = []

for threshold in THRESHOLDS:
    predictions = (score_series >= threshold).astype(int)

    pred_series = pd.Series(
        predictions,
        index=test_idx,
        name='prediction'
    )

    result = evaluate_predictions(
        run_name=f'threshold_{threshold}',
        y_test=y_test,
        pred_series=pred_series,
        meta_test=meta_test,
        threshold=threshold
    )

    threshold_results.append(result)

    predictions_df = df.loc[test_idx].copy()
    predictions_df['real_value'] = y_test
    predictions_df['prediction'] = pred_series
    predictions_df['anomaly_score'] = score_series
    predictions_df['threshold'] = threshold

    safe_threshold_name = str(threshold).replace('.', '_')

    predictions_df.to_csv(
        f'predictions_threshold_{safe_threshold_name}.csv',
        index=False
    )

    # Keep evaluate.py working with the default 0.35 threshold.
    if threshold == DEFAULT_THRESHOLD:
        predictions_df.to_csv(
            'predictions.csv',
            index=False
        )


# ---------------------------------------------------------------------
# Save feature importance
# ---------------------------------------------------------------------

if hasattr(model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X_full.columns,
        'importance': model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        by='importance',
        ascending=False
    )

    feature_importance.to_csv(
        'feature_importance.csv',
        index=False
    )


# ---------------------------------------------------------------------
# Save and print threshold results
# ---------------------------------------------------------------------

threshold_df = pd.DataFrame(threshold_results)
threshold_df.to_csv('threshold_results.csv', index=False)

print('\nTHRESHOLD RESULTS')
print(
    threshold_df[
        [
            'threshold',
            'accuracy',
            'precision',
            'f1',
            'false_positives',
            'false_negatives',
            'anomaly_recall',
            'oscillation_recall',
            'current_osc_recall',
            'voltage_osc_recall',
        ]
    ]
)