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



# sensors = ['temperature', 'pressure', 'vibration', 'flow_rate', 'voltage', 'current']
# anomalies = ['drift', 'oscillation']

# for i, row in df.iterrows():
#     if row['real_value'] == 1:
#         sensor = row['target_sensor']
        
#         df.at[i, 'target_5_step_diff'] = row[f'{sensor}_5_step_diff']
#         df.at[i, 'target_10_step_diff'] = row[f'{sensor}_10_step_diff']
#         df.at[i, 'target_short_long_mean_diff'] = row[f'{sensor}_short_long_mean_diff']
#         if row['prediction'] == 0 and row['anomaly_type'] == 'drift':
#                     print(sensor)
#                     print(row[f'{sensor}_5_step_diff'])
#                     print(row[f'{sensor}_10_step_diff'])
#                     print(row[f'{sensor}_short_long_mean_diff'])  

# print('Correct drift rows:')
# print(correct_drift_rows[
#     ['target_5_step_diff', 'target_10_step_diff', 'target_short_long_mean_diff']
# ].describe())

# print('\nMissed drift rows:')
# print(missed_drift_rows[
#     ['target_5_step_diff', 'target_10_step_diff', 'target_short_long_mean_diff']
# ].describe())

# def build_sensor_comparisons(anomaly_rows, sensors):
#     by_sensor = {}
#     correct = {}
#     missed = {}
#     correct_compare = {}
#     missed_compare = {}

#     for sensor in sensors:
#         by_sensor[sensor] = anomaly_rows[anomaly_rows['target_sensor'] == sensor]

#         correct[sensor] = by_sensor[sensor][by_sensor[sensor]['prediction'] == 1]
#         missed[sensor] = by_sensor[sensor][by_sensor[sensor]['prediction'] == 0]

#         compare_cols = [
#             f'{sensor}_5_step_diff',
#             f'{sensor}_10_step_diff',
#             f'{sensor}_short_long_mean_diff',
#             f'{sensor}_roll_std',
#             f'{sensor}_roll_mean'
#         ]

#         correct_compare[sensor] = correct[sensor][compare_cols].agg(
#             ['mean', 'median', 'std', 'min', 'max']
#         ).T

#         missed_compare[sensor] = missed[sensor][compare_cols].agg(
#             ['mean', 'median', 'std', 'min', 'max']
#         ).T

#     return by_sensor, correct, missed, correct_compare, missed_compare

# results = {}

# results['drift'] = {}
# results['drift']['by_sensor'], results['drift']['correct'], results['drift']['missed'], \
# results['drift']['correct_compare'], results['drift']['missed_compare'] = \
#     build_sensor_comparisons(drift_rows, sensors)

# results['oscillation'] = {}
# results['oscillation']['by_sensor'], results['oscillation']['correct'], results['oscillation']['missed'], \
# results['oscillation']['correct_compare'], results['oscillation']['missed_compare'] = \
#     build_sensor_comparisons(osc_rows, sensors)
    
# for anomaly in anomalies:
#     print(anomaly)
#     for sensor in sensors:
#         print(sensor)
#         print(results[anomaly]['correct_compare'][sensor])
#         print(results[anomaly]['missed_compare'][sensor])
#         print('-----------------------------------')
#     print('-----------------------------------')

# correct_drift_rows = df[
#     (df['anomaly_type'] == 'drift') & 
#     (df['real_value'] == 1) & 
#     (df['prediction'] == 1)
#     ]

# missed_drift_rows = df[
#     (df['anomaly_type'] == 'drift') & 
#     (df['real_value'] == 1) & 
#     (df['prediction'] == 0)
#     ]

# df['target_5_step_diff'] = float('nan')
# df['target_10_step_diff'] = float('nan')
# df['target_short_long_mean_diff'] = float('nan') 