from sklearn import metrics
import pandas as pd

df = pd.read_csv('predictions.csv')
y_test = df['real_value'].tolist()
predictions = df['prediction'].tolist()

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