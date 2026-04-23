from sklearn import metrics
import pandas as pd

df = pd.read_csv('predictions.csv')
y_test = df['real_value'].tolist()
predictions = df['prediction'].tolist()
counts = pd.read_csv('feature_row_retention.csv')
            
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
        
num_spike = 0
num_spike_correct = 0
num_drop = 0
num_drop_correct = 0
num_drift = 0
num_drift_correct = 0
num_oscillation = 0
num_oscillation_correct = 0
num_stuck_sensor = 0
num_stuck_sensor_correct = 0
num_impossible = 0
num_impossible_correct = 0

for i, row in df.iterrows():
    if row['real_value'] == 1:
        if row['anomaly_type'] == 'spike':
            num_spike += 1
            if row['prediction'] == 1:
                num_spike_correct += 1
        elif row['anomaly_type'] == 'drop':
            num_drop += 1
            if row['prediction'] == 1:
                num_drop_correct += 1
        elif row['anomaly_type'] == 'oscillation':
            num_oscillation += 1
            if row['prediction'] == 1:
                num_oscillation_correct += 1
        elif row['anomaly_type'] == 'drift':
            num_drift += 1
            if row['prediction'] == 1:
                num_drift_correct += 1
        elif row['anomaly_type'] == 'stuck_sensor':
            num_stuck_sensor += 1
            if row['prediction'] == 1:
                num_stuck_sensor_correct += 1
        elif row['anomaly_type'] == 'impossible_value':
            num_impossible += 1
            if row['prediction'] == 1:
                num_impossible_correct += 1
                
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

#Row-retention table
print(counts)

#Overall classification summary
print(metrics.confusion_matrix(y_test, predictions))
print(metrics.accuracy_score(y_test, predictions))

#Per-anomaly summary
print("spike: ", num_spike_correct, num_spike, num_spike_correct / num_spike)
print("drop: ", num_drop_correct, num_drop, num_drop_correct / num_drop)
print("drift: ", num_drift_correct, num_drift, num_drift_correct / num_drift)
print("oscillation: ", num_oscillation_correct, num_oscillation, num_oscillation_correct / num_oscillation)
print("stuck_sensor: ", num_stuck_sensor_correct, num_stuck_sensor, num_stuck_sensor_correct / num_stuck_sensor)
print("impossible_value: ", num_impossible_correct, num_impossible, num_impossible_correct / num_impossible)

#Drift/Oscillation by sensor
print(drift_by_sensor)
print(osc_by_sensor)

#current osc correct vs missed
debug_oscillation_sensor('current')
debug_oscillation_sensor('voltage')