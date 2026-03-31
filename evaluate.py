from sklearn import metrics
import pandas as pd

sensors = ['temperature', 'pressure', 'vibration', 'flow_rate', 'voltage', 'current']

df = pd.read_csv('predictions.csv')
y_test = df['real_value'].tolist()
predictions = df['prediction'].tolist()

df['target_5_step_diff'] = float('nan')
df['target_10_step_diff'] = float('nan')
df['target_short_long_mean_diff'] = float('nan')

for i, row in df.iterrows():
    if row['real_value'] == 1:
        sensor = row['target_sensor']
        
        df.at[i, 'target_5_step_diff'] = row[f'{sensor}_5_step_diff']
        df.at[i, 'target_10_step_diff'] = row[f'{sensor}_10_step_diff']
        df.at[i, 'target_short_long_mean_diff'] = row[f'{sensor}_short_long_mean_diff']
        
        if row['prediction'] == 0 and row['anomaly_type'] == 'drift':
            print(sensor)
            print(row[f'{sensor}_5_step_diff'])
            print(row[f'{sensor}_10_step_diff'])
            print(row[f'{sensor}_short_long_mean_diff']) 
            
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

print(drift_by_sensor)

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

print(osc_by_sensor)
    
correct_drift_rows = df[
    (df['anomaly_type'] == 'drift') & 
    (df['real_value'] == 1) & 
    (df['prediction'] == 1)
    ]

missed_drift_rows = df[
    (df['anomaly_type'] == 'drift') & 
    (df['real_value'] == 1) & 
    (df['prediction'] == 0)
    ]

print('Correct drift rows:')
print(correct_drift_rows[
    ['target_5_step_diff', 'target_10_step_diff', 'target_short_long_mean_diff']
].describe())

print('\nMissed drift rows:')
print(missed_drift_rows[
    ['target_5_step_diff', 'target_10_step_diff', 'target_short_long_mean_diff']
].describe())
        
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

print(metrics.accuracy_score(y_test, predictions))
print(metrics.confusion_matrix(y_test, predictions))
print("spike: ", num_spike, num_spike_correct)
print("drop: ", num_drop, num_drop_correct)
print("drift: ", num_drift, num_drift_correct)
print("oscillation: ", num_oscillation, num_oscillation_correct)
print("stuck_sensor: ", num_stuck_sensor, num_stuck_sensor_correct)
print("impossible_value: ", num_impossible, num_impossible_correct)