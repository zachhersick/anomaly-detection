import pandas as pd

sensors = ['temperature', 'pressure', 'vibration', 'flow_rate', 'voltage', 'current']

rows_df = pd.read_csv('sensor_data_raw.csv')
df = rows_df.sort_values(by=['machine_id', 'step']).reset_index(drop=True)
grouped = df.groupby('machine_id')

window = 10
long_window = 50

for sensor in sensors:
    df[f'{sensor}_delta'] = grouped[sensor].diff()
    df[f'{sensor}_abs_delta'] = grouped[sensor].diff().abs()
    df[f'{sensor}_roll_mean'] = grouped[sensor].rolling(window).mean().reset_index(level=0, drop=True)
    df[f'{sensor}_long_roll_mean'] = grouped[sensor].rolling(long_window).mean().reset_index(level=0, drop=True)
    df[f'{sensor}_roll_std'] = grouped[sensor].rolling(window).std().reset_index(level=0, drop=True)
    df[f'{sensor}_roll_min'] = grouped[sensor].rolling(window).min().reset_index(level=0, drop=True)
    df[f'{sensor}_roll_max'] = grouped[sensor].rolling(window).max().reset_index(level=0, drop=True)
    df[f'{sensor}_dev'] = df[sensor] - df[f'{sensor}_roll_mean']
    df[f'{sensor}_zscore'] = (
        (df[sensor] - df[f'{sensor}_roll_mean']) /
        df[f'{sensor}_roll_std']
    )
    df[f'{sensor}_5_step_diff'] = grouped[sensor].diff(5)
    df[f'{sensor}_10_step_diff'] = grouped[sensor].diff(10)
    df[f'{sensor}_roll_avg_delta'] = grouped[f'{sensor}_delta'].rolling(window).mean().reset_index(level=0, drop=True)
    df[f'{sensor}_short_long_mean_diff'] = df[f'{sensor}_roll_mean'] - df[f'{sensor}_long_roll_mean']

df = df.dropna()
df.to_csv('sensor_data_features.csv', index=False)