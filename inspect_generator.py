import pandas as pd
import matplotlib.pyplot as plt

def build_segments(df, anomaly_type, target_sensor):
    filtered = (
        df[
            (df['anomaly_type'] == anomaly_type) &
            (df['target_sensor'] == target_sensor)
        ]
        .sort_values(['machine_id', 'step'])
        .reset_index(drop=True)
        .copy()
    )
    
    filtered['segment_id'] = (
        (filtered['machine_id'] != filtered['machine_id'].shift(1)) |
        (filtered['step'] != filtered['step'].shift(1) + 1)
    ).cumsum()
    
    summary = filtered.groupby(['machine_id', 'segment_id']).agg(
        start_step=('step', 'min'),
        end_step=('step', 'max'),
        length=('step', 'size')
    )
    
    return filtered, summary

def plot_segments(segmented_df, value_col):
    for (machine_id, segment_id), segment_df in segmented_df.groupby(['machine_id', 'segment_id']):
        plt.figure()
        plt.plot(segment_df['step'], segment_df[value_col])
        plt.title(f'machine {machine_id}, segment {segment_id}')
        plt.show()

df = pd.read_csv('sensor_data_raw.csv')

for sensor in ['temperature', 'voltage', 'current']:
    filtered, summary = build_segments(df, 'drift', sensor)
    print(f'\n{sensor.upper()} DRIFT')
    print(summary)
    plot_segments(filtered, sensor)