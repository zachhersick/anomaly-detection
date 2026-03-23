import pandas as pd

df = pd.read_csv('sensor_data_features.csv')

meta_cols = ['step', 'machine_id', 'target_sensor']
label_cols = [
            'any_anomaly', 'anomaly_type', 'temperature_anomaly', 
            'vibration_anomaly', 'flow_rate_anomaly', 'pressure_anomaly', 
            'current_anomaly', 'voltage_anomaly'
            ]

y = df['any_anomaly']
X = df.drop(columns=meta_cols + label_cols)

print(X.columns)
print(X.shape)
print(y.value_counts())

