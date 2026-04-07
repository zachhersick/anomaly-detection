import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

med_weights = ['spike', 'drop', 'stuck_sensor', 'impossible_value']

def build_sample_weights(training_rows):
    weights = []
    for i, row in training_rows.iterrows():
        if row['anomaly_type'] == 'none':
            weights.append(1)
        elif row['anomaly_type'] in med_weights:
            weights.append(1)
        elif row['anomaly_type'] == 'oscillation':
            weights.append(2)
        elif row['anomaly_type'] == 'drift':
            weights.append(3)
        else:
            raise ValueError('bad label')
    return weights

def summarize_weights(training_rows, weights):
    tr_copy = training_rows.copy()
    tr_copy['weight'] = weights
    grouped = tr_copy.groupby('anomaly_type')
    for anomaly_type, df in grouped:
        print(f"anomaly_type: {anomaly_type}")
        print(f"rows: {len(df)}")
        print(f"weights: {df['weight'].unique()}")
        print('------------------')

df = pd.read_csv('sensor_data_features.csv')

meta_cols = ['step', 'machine_id', 'target_sensor']
label_cols = [
            'any_anomaly', 'anomaly_type', 'temperature_anomaly', 
            'vibration_anomaly', 'flow_rate_anomaly', 'pressure_anomaly', 
            'current_anomaly', 'voltage_anomaly'
            ]

metadata = df[['anomaly_type']].copy()

y = df['any_anomaly']
X = df.drop(columns=meta_cols + label_cols)

split_index = int(0.8*len(df))

X_train = X[:split_index]
y_train = y[:split_index]

X_test = X[split_index:]
y_test = y[split_index:]

meta_train = metadata[:split_index]
meta_test = metadata[split_index:]

weights = build_sample_weights(meta_train)
print(len(X_train))
print(len(weights))
summarize_weights(meta_train, weights)

df_test = df.iloc[split_index:].copy()

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train, sample_weight=weights)

importances_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})
importances_df = importances_df.sort_values(
    by='importance',
    ascending=False
)

importances_df.to_csv('feature_importance.csv', index=False)
predictions = model.predict(X_test)

df_test['real_value'] = y_test
df_test['prediction'] = predictions

df_test.to_csv('predictions.csv', index=False)