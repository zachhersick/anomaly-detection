import pandas as pd
import numpy as np

sensors = ['temperature', 'pressure', 'vibration', 'flow_rate', 'voltage', 'current']

rows_df = pd.read_csv('sensor_data_raw.csv')
df = rows_df.sort_values(by=['machine_id', 'step']).reset_index(drop=True)
grouped = df.groupby('machine_id')

window = 10
long_window = 50

def slope(window_values):
    x = np.arange(len(window_values))
    return np.polyfit(x, window_values, 1)[0]

for sensor in sensors:
    # df[f'{sensor}_zscore_undefined'] = 0
    # df[f'{sensor}_variability_undefined'] = 0
    df[f'{sensor}_delta'] = grouped[sensor].diff()
    df[f'{sensor}_abs_delta'] = df[f'{sensor}_delta'].abs()

    delta_sign = np.sign(df[f'{sensor}_delta'])
    prev_sign = delta_sign.groupby(df['machine_id']).shift(1)
    
    new_run = (
        prev_sign.isna() |
        (delta_sign == 0) |
        (delta_sign != prev_sign)
    )
    
    run_id = new_run.groupby(df['machine_id']).cumsum()
    
    same_dir_run = df.groupby([df['machine_id'], run_id]).cumcount() + 1
    same_dir_run[delta_sign == 0] = 0
    
    df[f'{sensor}_same_dir_run'] = same_dir_run
    df[f'{sensor}_same_dir_run_10'] = same_dir_run.clip(upper=10)
    
    sign_flip = (
        delta_sign * prev_sign < 0
        ).astype(int)
    df[f'{sensor}_sign_change_count'] = (
        sign_flip.groupby(df['machine_id'])
        .rolling(window)
        .sum()
        .reset_index(level=0, drop=True)
    )
    
    df[f'{sensor}_roll_mean'] = grouped[sensor].rolling(window).mean().reset_index(level=0, drop=True)
    df[f'{sensor}_long_roll_mean'] = grouped[sensor].rolling(long_window).mean().reset_index(level=0, drop=True)
    df[f'{sensor}_roll_std'] = grouped[sensor].rolling(window).std().reset_index(level=0, drop=True)
    df[f'{sensor}_long_roll_std'] = grouped[sensor].rolling(long_window).std().reset_index(level=0, drop=True)
    df[f'{sensor}_roll_min'] = grouped[sensor].rolling(window).min().reset_index(level=0, drop=True)
    df[f'{sensor}_roll_max'] = grouped[sensor].rolling(window).max().reset_index(level=0, drop=True)
    df[f'{sensor}_dev'] = df[sensor] - df[f'{sensor}_roll_mean']
    
    zscore_denom = df[f'{sensor}_roll_std'].where(df[f'{sensor}_roll_std'] > 1e-6, float('nan'))
    
    df[f'{sensor}_zscore'] = ((df[sensor] - df[f'{sensor}_roll_mean']) / zscore_denom)
    df[f'{sensor}_zscore']= df[f'{sensor}_zscore'].fillna(0)
    # if df[f'{sensor}_zscore'] == 0:
    #     df[f'{sensor}_zscore_undefined'] = 1
    
    df[f'{sensor}_5_step_diff'] = grouped[sensor].diff(5)
    df[f'{sensor}_10_step_diff'] = grouped[sensor].diff(10)
    df[f'{sensor}_roll_avg_delta'] = df.groupby('machine_id')[f'{sensor}_delta'].rolling(window).mean().reset_index(level=0, drop=True)
    df[f'{sensor}_short_long_mean_diff'] = df[f'{sensor}_roll_mean'] - df[f'{sensor}_long_roll_mean']
    df[f'{sensor}_abs_5_step_diff'] = grouped[sensor].diff(5).abs()
    df[f'{sensor}_abs_10_step_diff'] = grouped[sensor].diff(10).abs()
    df[f'{sensor}_roll_range'] = df[f'{sensor}_roll_max'] - df[f'{sensor}_roll_min']
    
    df[f'{sensor}_roll_mean_abs_delta'] = df.groupby('machine_id')[f'{sensor}_abs_delta'].rolling(window).mean().reset_index(level=0, drop=True)
    
    var_denom = df[f'{sensor}_long_roll_std'].where(df[f'{sensor}_long_roll_std'] > 1e-6, float('nan'))
    
    df[f'{sensor}_variability'] = df[f'{sensor}_roll_std'] / var_denom
    df[f'{sensor}_variability']= df[f'{sensor}_variability'].fillna(0)
    # if df[f'{sensor}_variability'] == 0:
    #     df[f'{sensor}_variability_undefined'] = 1
    
    df[f'{sensor}_slope_10'] = (
        df.groupby('machine_id')[sensor]
        .rolling(10)
        .apply(slope, raw=True)
        .reset_index(level=0, drop=True)
    )
    
    df[f'{sensor}_slope_25'] = (
        df.groupby('machine_id')[sensor]
        .rolling(25)
        .apply(slope, raw=True)
        .reset_index(level=0, drop=True)
    )
    
    df[f'{sensor}_cum_change_25'] = grouped[sensor].diff(25)
    df[f'{sensor}_long_base_dev'] = df[sensor] - df[f'{sensor}_long_roll_mean']
    # df[f'{sensor}_slope_ratio'] = df[f'{sensor}_slope_10'] / df[f'{sensor}_slope_25']

before_counts = df.groupby(['anomaly_type', 'target_sensor']).size().rename('before_drop')

df = df.dropna()

after_counts = df.groupby(['anomaly_type', 'target_sensor']).size().rename('after_drop')

counts = pd.concat([before_counts, after_counts], axis=1).fillna(0)
counts['before_drop'] = counts['before_drop'].astype(int)
counts['after_drop'] = counts['after_drop'].astype(int)
counts['rows_dropped'] = counts['before_drop'] - counts['after_drop']
counts['pct_kept'] = counts['after_drop'] / counts['before_drop']

counts = counts.reset_index()

df.to_csv('sensor_data_features.csv', index=False)
counts.to_csv('feature_row_retention.csv', index=False)