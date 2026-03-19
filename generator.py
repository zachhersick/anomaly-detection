import random

anomalies = ['spike', 'drop', 'drift', 'oscillation', 'stuck_sensor', 'impossible_value']
sensor_ranges = {'temperature': (60,100), 'pressure': (30,80), 'vibration': (0,10), 'flow_rate': (0.5,2), 'voltage': (110,130), 'current': (5,20)}
sensor_noise = {'temperature': (0,2), 'pressure': (0,1), 'vibration': (0,0.1), 'flow_rate': (0,0.05), 'voltage': (0.5,2), 'current': (0,2)}
anomaly_probability = 0.05
anomaly_duration = {'spike': (1,3), 'drop': (1,3), 'drift': (50,100), 'oscillation': (30,60), 'stuck_sensor': (50,100), 'impossible_value': (1,2)}
random_seed = random.randint(1, 10000)
random.seed(random_seed)
num_machines = 3
machines = []

for i in range(num_machines):
    machines.append({'current_sensor_values': {
                        'temperature': random.uniform(sensor_ranges['temperature'][0], sensor_ranges['temperature'][1]),
                        'pressure': random.uniform(sensor_ranges['pressure'][0], sensor_ranges['pressure'][1]),
                        'vibration': random.uniform(sensor_ranges['vibration'][0], sensor_ranges['vibration'][1]),
                        'flow_rate': random.uniform(sensor_ranges['flow_rate'][0], sensor_ranges['flow_rate'][1]),
                        'voltage': random.uniform(sensor_ranges['voltage'][0], sensor_ranges['voltage'][1]),
                        'current': random.uniform(sensor_ranges['current'][0], sensor_ranges['current'][1])
                        }, 
                  'mode': "NORMAL", 
                  'anomaly_type': None, 
                  'remaining_duration': 0, 
                  'drift_rate': random.uniform(0.005, 0.02),
                  'drift_direction': random.choice([-1, 1])})

def step_function(previous_value, sensor):
    next_value = previous_value + random.uniform(-(sensor_noise[sensor][1]), sensor_noise[sensor][1])
    return next_value

def anomaly_spike(previous_value):
    next_value = previous_value + random.uniform(40, 80)
    return next_value
    
def anomaly_drop(previous_value):
    next_value = previous_value - random.uniform(40, 80)
    return next_value

def anomaly_drift(previous_value, machine, sensor):
    next_value = previous_value + (machine['drift_direction'] * machine['drift_rate']) + random.uniform(sensor_noise[sensor][0], sensor_noise[sensor][1])
    return next_value

num_timesteps = 300

for step in range(num_timesteps):
    for machine in machines:
        
        if machine['mode'] == 'NORMAL':
            if ((random.randint(1, 100)/100) <= anomaly_probability):
                machine['mode'] == 'ANOMALY'
                machine['anomaly_type'] = random.choice(anomalies)
                machine['remaining_duration'] = random.randint(int(anomaly_duration[machine['anomaly_type']][0]), int(anomaly_duration[machine['anomtaly_type']][1]))