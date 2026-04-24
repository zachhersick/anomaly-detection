from sklearn import metrics
import pandas as pd

df = pd.read_csv('predictions.csv')
y_test = df['real_value']
predictions = df['prediction']
counts = pd.read_csv('feature_row_retention.csv')

DEBUG_OSCILLATION_DETAILS = False

try:
    feature_importance = pd.read_csv('feature_importance.csv')
except FileNotFoundError:
    feature_importance = None

drift_rows = df[
    (df['real_value'] == 1) &
    (df['anomaly_type'] == 'drift')
]

drift_by_sensor = drift_rows.groupby(by='target_sensor').agg(
    total=('prediction', 'size'),
    correct=('prediction', lambda x: (x == 1).sum()),
    missed=('prediction', lambda x: (x == 0).sum())
)

drift_by_sensor['recall'] = drift_by_sensor['correct'] / drift_by_sensor['total']

osc_rows = df[
    (df['real_value'] == 1) &
    (df['anomaly_type'] == 'oscillation')
]

osc_by_sensor = osc_rows.groupby(by='target_sensor').agg(
    total=('prediction', 'size'),
    correct=('prediction', lambda x: (x == 1).sum()),
    missed=('prediction', lambda x: (x == 0).sum())
)

osc_by_sensor['recall'] = osc_by_sensor['correct'] / osc_by_sensor['total']

anomaly_rows = df[
    df['real_value'] == 1
]

anomaly_by_type = anomaly_rows.groupby(by='anomaly_type').agg(
    total=('prediction', 'size'),
    correct=('prediction', lambda x: (x == 1).sum()),
    missed=('prediction', lambda x: (x == 0).sum())
)

anomaly_by_type['recall'] = anomaly_by_type['correct'] / anomaly_by_type['total']

anomaly_by_sensor = anomaly_rows.groupby(by='target_sensor').agg(
    total=('prediction', 'size'),
    correct=('prediction', lambda x: (x == 1).sum()),
    missed=('prediction', lambda x: (x == 0).sum())
)

anomaly_by_sensor['recall'] = anomaly_by_sensor['correct'] / anomaly_by_sensor['total']

false_positives = df[
    (df['real_value'] == 0) &
    (df['prediction'] == 1)
]

false_negatives = df[
    (df['real_value'] == 1) &
    (df['prediction'] == 0)
]

osc_feature_suffixes = [
    'lag_5_autocorr',
    'lag_10_autocorr',
    'centered_zero_cross_count_10',
    'centered_zero_cross_count_20',
    'dir_imbalance_10',
    'dir_imbalance_20',
    'trend_ratio_10',
    'trend_ratio_25',
    'center_balance_10',
    'center_balance_20',
]

def debug_oscillation_sensor(sensor):
    sensor_rows = df.loc[
        (df['real_value'] == 1) &
        (df['anomaly_type'] == 'oscillation') &
        (df['target_sensor'] == sensor)
    ]

    correct_rows = sensor_rows.loc[sensor_rows['prediction'] == 1]
    missed_rows = sensor_rows.loc[sensor_rows['prediction'] == 0]

    compare_cols = [
        f'{sensor}_{suffix}'
        for suffix in osc_feature_suffixes
        if f'{sensor}_{suffix}' in df.columns
    ]

    if not compare_cols:
        print(f'\n--- OSCILLATION DEBUG: {sensor} ---')
        print('No matching oscillation feature columns found.')
        return

    print(f'\n--- OSCILLATION DEBUG: {sensor} ---')
    print('correct:', len(correct_rows))
    print('missed:', len(missed_rows))
    print('columns:', compare_cols)

    print('\ncorrect describe')
    print(correct_rows[compare_cols].describe())

    print('\nmissed describe')
    print(missed_rows[compare_cols].describe())

    print('\nmean difference (correct - missed)')
    print(correct_rows[compare_cols].mean() - missed_rows[compare_cols].mean())

    print('\ncorrect NaN counts')
    print(correct_rows[compare_cols].isna().sum())

    print('\nmissed NaN counts')
    print(missed_rows[compare_cols].isna().sum())


# Row-retention table
print('\n==============================')
print('ROW RETENTION')
print('==============================')
print(counts)


# Overall classification summary
print('\n==============================')
print('FINAL MODEL EVALUATION')
print('==============================')

if 'threshold' in df.columns:
    print('threshold:', df['threshold'].iloc[0])

confusion_matrix = metrics.confusion_matrix(y_test, predictions, labels=[0, 1])
tn, fp, fn, tp = confusion_matrix.ravel()

print('\nConfusion Matrix')
print(confusion_matrix)

print('\nOverall Metrics')
print('accuracy:', metrics.accuracy_score(y_test, predictions))
print('precision:', metrics.precision_score(y_test, predictions, zero_division=0))
print('recall:', metrics.recall_score(y_test, predictions, zero_division=0))
print('f1:', metrics.f1_score(y_test, predictions, zero_division=0))
print('false positives:', fp)
print('false negatives:', fn)
print('true positives:', tp)
print('true negatives:', tn)


# Per-anomaly summary
print('\nRecall by Anomaly Type')
print(anomaly_by_type.sort_values(by='recall'))


# Target sensor summary
print('\nRecall by Target Sensor')
print(anomaly_by_sensor.sort_values(by='recall'))


# Drift/Oscillation by sensor
print('\nDrift Recall by Target Sensor')
print(drift_by_sensor.sort_values(by='recall'))

print('\nOscillation Recall by Target Sensor')
print(osc_by_sensor.sort_values(by='recall'))


# False positive / false negative summaries
print('\nFalse Positive Count:', len(false_positives))
print('False Negative Count:', len(false_negatives))

print('\nFalse Negatives by Anomaly Type')
print(false_negatives['anomaly_type'].value_counts())

print('\nFalse Negatives by Target Sensor')
print(false_negatives['target_sensor'].value_counts())


# Feature importance
if feature_importance is not None:
    print('\nTop 20 Feature Importances')
    print(feature_importance.head(20))
else:
    print('\nNo feature_importance.csv found.')


# Optional current/voltage oscillation debug
if DEBUG_OSCILLATION_DETAILS:
    debug_oscillation_sensor('current')
    debug_oscillation_sensor('voltage')