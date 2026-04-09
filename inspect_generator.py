import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('sensor_data_raw.csv')

current_drift_df = df[
    (df['anomaly_type'] == 'drift') &
    (df['target_sensor'] == 'current')
].copy()

current_drift_df = current_drift_df.sort_values(
    ['machine_id', 'step']
).reset_index(drop=True)

current_drift_df['prev_machine'] = current_drift_df['machine_id'].shift(1)
current_drift_df['prev_step'] = current_drift_df['step'].shift(1)

current_drift_df['new_segment'] = (
    (current_drift_df['machine_id'] != current_drift_df['prev_machine']) |
    (current_drift_df['step'] != current_drift_df['prev_step'] + 1)
)

current_drift_df['segment_id'] = current_drift_df['new_segment'].cumsum()

segment_summary = current_drift_df.groupby(['machine_id', 'segment_id']).agg(
    start_step=('step', 'min'),
    end_step=('step', 'max'),
    length=('step', 'size')
)

print(segment_summary)

grouped = current_drift_df.groupby(['machine_id', 'segment_id'])

for (machine_id, segment_id), segment_df in grouped:
    plt.figure()
    plt.plot(segment_df['step'], segment_df['current'])
    plt.title(f'machine {machine_id}, segment {segment_id}')
    plt.show()

# temp_drift_df = df[
#     (df['anomaly_type'] == 'drift') &
#     (df['target_sensor'] == 'temperature')
# ]

# volt_drift_df = df[
#     (df['anomaly_type'] == 'drift') &
#     (df['target_sensor'] == 'voltage')
# ]

# volt_osc_df = df[
#     (df['anomaly_type'] == 'oscillation') &
#     (df['target_sensor'] == 'voltage')
# ]

# current_drift_df_1 = current_drift_df[
#     (current_drift_df['machine_id'] == 3)
# ]