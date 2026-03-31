import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

df = pd.read_csv('sensor_data_features.csv')

meta_cols = ['step', 'machine_id', 'target_sensor']
label_cols = [
            'any_anomaly', 'anomaly_type', 'temperature_anomaly', 
            'vibration_anomaly', 'flow_rate_anomaly', 'pressure_anomaly', 
            'current_anomaly', 'voltage_anomaly'
            ]

y = df['any_anomaly']
X = df.drop(columns=meta_cols + label_cols)

split_index = int(0.8*len(df))

X_train = X[:split_index]
y_train = y[:split_index]

X_test = X[split_index:]
y_test = y[split_index:]

df_test = df.iloc[split_index:].copy()

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

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

df_test.to_csv('predictions.csv')