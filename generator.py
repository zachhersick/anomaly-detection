import random, statistics

anomalies = ['spike', 'drop', 'drift', 'oscillation', 'stuck_sensor', 'impossible_value']
sensors = ['temperature', 'pressure', 'vibration', 'flow_rate', 'voltage', 'current']
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
                  'is_anomaly': False,
                  'remaining_duration': 0, 
                  'drift_rate': random.uniform(0.005, 0.02),
                  'drift_direction': None})

def step_function(previous_value, sensor):
    next_value = previous_value + random.uniform(-(sensor_noise[sensor][1]), sensor_noise[sensor][1])
    return next_value

def anomaly_spike(previous_value, sensor):
    next_value = random.uniform(2*sensor_ranges[sensor][0], 2.33*sensor_ranges[sensor][1])
    return next_value
    
def anomaly_drop(previous_value, sensor):
    next_value = random.uniform(0.33*sensor_ranges[sensor][0], 0.66*sensor_ranges[sensor][1])
    return next_value

def anomaly_drift(previous_value, machine, sensor):
    next_value = previous_value + (machine['drift_direction'] * machine['drift_rate']) + random.uniform(-(sensor_noise[sensor][1]), sensor_noise[sensor][1])
    return next_value

def anomaly_oscillation(previous_value, previous_sign, sensor, center, amplitude):
    if previous_sign == '+':
        next_value = center - amplitude
    else:
        next_value = center + amplitude
    return next_value

def anomaly_stuck_sensor(previous_value):
    return previous_value

def anomaly_impossible_value(sensor):
    if sensor == 'temperature':
        next_value = random.choice(random.uniform(-10000, -100), random.uniform(1000, 100000))
    if sensor == 'pressure' or sensor == 'vibration' or sensor == 'flow_rate':
        next_value = random.uniform(-1000, -1)
    if sensor == 'voltage':
        next_value = 0
    if sensor == 'current':
        next_value = random.choice(random.uniform(-10000, -100), random.uniform(1000, 100000))
    return next_value

num_timesteps = 300

for step in range(num_timesteps):
    for i, machine in enumerate(machines):
        machine_id = i
        
        if machine['mode'] == 'NORMAL':
            if (random.random() <= anomaly_probability):
                machine['mode'] = 'ANOMALY'
                machine['is_anomaly'] = True
                machine['anomaly_type'] = random.choice(anomalies)
                machine['remaining_duration'] = int(random.randint(int(anomaly_duration[str(machine['anomaly_type'])][0]), int(anomaly_duration[str(machine['anomaly_type'])][1])))
                if machine['anomaly_type'] == 'drift':
                    machine['drift_direction'] = random.choice([-1, 1])
            
            if machine['mode'] == 'NORMAL':
                for sensor in sensors:
                    machine['current_sensor_values'][sensor] = step_function(machine['current_sensor_values'][sensor], sensor)
                    
        if machine['mode'] == 'ANOMALY':
            for sensor in sensors:
                if machine['anomaly_type'] == 'spike':
                    machine['current_sensor_values'][sensor] = anomaly_spike(machine['current_sensor_values'][sensor], sensor)
                if machine['anomaly_type'] == 'drop':
                    machine['current_sensor_values'][sensor] = anomaly_drop(machine['current_sensor_values'][sensor], sensor)
                if machine['anomaly_type'] == 'drift':
                    machine['current_sensor_values'][sensor] = anomaly_drift(machine['current_sensor_values'][sensor], machine, sensor)
                # if machine['anomaly_type'] == 'oscillation':
                #     machine[sensor] = anomaly_spike(machine[sensor])
                # if machine['anomaly_type'] == 'stuck_sensor':
                #     machine[sensor] = anomaly_spike(machine[sensor])
                # if machine['anomaly_type'] == 'impossible_value':
                #     machine[sensor] = anomaly_spike(machine[sensor])
            if (machine['remaining_duration'] > 0):
                machine['remaining_duration'] -= 1
            
            if machine['remaining_duration'] == 0:
                machine['mode'] = 'NORMAL'
                machine['anomaly_type'] = None
                machine['drift_direction'] = None
                machine['is_anomaly'] = False
        
        print(step, machine_id, machine['current_sensor_values']['temperature'], machine['current_sensor_values']['pressure'], machine['mode'], machine['anomaly_type'])
                