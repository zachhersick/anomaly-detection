import random, math, statistics
import pandas as pd

rows = []

anomalies = ['spike', 'drop', 'drift', 'oscillation', 'stuck_sensor', 'impossible_value']
sensors = ['temperature', 'pressure', 'vibration', 'flow_rate', 'voltage', 'current']

sensor_ranges = {
    'temperature': (60,100), 
    'pressure': (30,80), 
    'vibration': (0,10), 
    'flow_rate': (0.5,2), 
    'voltage': (110,130), 
    'current': (5,20)
    }

sensor_noise = {
    'temperature': (0,2), 
    'pressure': (0,1), 
    'vibration': (0,0.1), 
    'flow_rate': (0,0.05), 
    'voltage': (0.5,2), 
    'current': (0,2)
    }

anomaly_probability = 0.01
anomaly_duration = {
    'spike': (1,3), 
    'drop': (1,3), 
    'drift': (50,100), 
    'oscillation': (30,60), 
    'stuck_sensor': (50,100), 
    'impossible_value': (1,2)
    }

seed = random.randint(1, 10000)
random.seed(seed)

# creates and returns a machine state independently of the current machine states
def init_machine():
    return {
        'values': {s: random.uniform(*sensor_ranges[s]) for s in sensors},
        'mode': "NORMAL", 
        'anomaly_type': 'none', 
        'remaining_duration': 0,
        'target_sensor': 'none', 
        'is_anomaly': {s: 0 for s in sensors},
        
        'drift_rate': random.uniform(0.02, 0.05),
        'drift_direction': 1,
        'stuck_value': None,
        'osc_center': None,
        'osc_amplitude': None,
        'osc_phase': 0
    }
    
#instantiate machines
num_machines = 10
machines = [init_machine() for i in range(num_machines)]

#normal data step function
def step_normal(previous_value, sensor):
    lo, hi = sensor_ranges[sensor]
    center = (lo + hi) / 2
    noise = random.uniform(-sensor_noise[sensor][1], sensor_noise[sensor][1])
    return previous_value + 0.1*(center - previous_value) + noise

#anomaly data step functions
def spike(previous_value, sensor):
    lo, hi = sensor_ranges[sensor]
    return previous_value + random.uniform(0.5*(hi-lo), (hi-lo))
    
def drop(previous_value, sensor):
    lo, hi = sensor_ranges[sensor]
    return previous_value - random.uniform(0.5*(hi-lo), (hi-lo))

def drift(previous_value, machine, sensor):
    noise = random.uniform(-(sensor_noise[sensor][1]), sensor_noise[sensor][1])
    return previous_value + machine['drift_direction'] * machine['drift_rate'] + noise

def oscillation(machine, sensor):
    noise = random.uniform(-(sensor_noise[sensor][1]), sensor_noise[sensor][1])
    machine['osc_phase'] += 0.3
    return machine['osc_center'] + machine['osc_amplitude'] * math.sin(machine['osc_phase']) + noise

def stuck_sensor(machine):
    return machine['stuck_value']

def impossible_value(sensor):
    lo, hi = sensor_ranges[sensor]
    if sensor in ['pressure', 'vibration', 'flow_rate']:
        return random.uniform(-10, -1)
    if sensor == 'voltage':
        return 0
    return random.uniform(hi*1.5, hi*3)

def clip(sensor, value):
    lo, hi = sensor_ranges[sensor]
    return max(lo * 0.5, min(hi * 1.5, value))

num_timesteps = 5000

#data loop
for step in range(num_timesteps):
    for i, machine in enumerate(machines):
        
        if machine['mode'] == 'NORMAL' and random.random() <= anomaly_probability:
            machine['mode'] = 'ANOMALY'
            machine['target_sensor'] = random.choice(sensors)
            machine['anomaly_type'] = random.choice(anomalies)
            machine['remaining_duration'] = random.randint(*anomaly_duration[machine['anomaly_type']])
            
            sensor = machine['target_sensor']
            
            if machine['anomaly_type'] == 'drift':
                machine['drift_direction'] = random.choice([-1, 1])
                machine['drift_rate'] = random.uniform(0.01, 0.1)
            
            if machine['anomaly_type'] == 'stuck_sensor':
                machine['stuck_value'] = machine['values'][sensor]
                
            if machine['anomaly_type'] == 'oscillation':
                lo, hi = sensor_ranges[sensor]
                machine['osc_center'] = machine['values'][sensor]
                machine['osc_amplitude'] = 0.2*(hi-lo)
                machine['osc_phase'] = 0
                    
        for sensor in sensors:
            if machine['mode'] == 'ANOMALY' and sensor == machine['target_sensor']:
                machine['is_anomaly'][sensor] = 1
                
                atype = machine['anomaly_type']
                if atype == 'spike':
                    machine['values'][sensor] = spike(machine['values'][sensor], sensor)
                    machine['values'][sensor] = clip(sensor, machine['values'][sensor])
                elif atype == 'drop':
                    machine['values'][sensor] = drop(machine['values'][sensor], sensor)
                    machine['values'][sensor] = clip(sensor, machine['values'][sensor])
                elif atype == 'drift':
                    machine['values'][sensor] = drift(machine['values'][sensor], machine, sensor)
                    machine['values'][sensor] = clip(sensor, machine['values'][sensor])
                elif atype == 'oscillation':
                    machine['values'][sensor] = oscillation(machine, sensor)
                    machine['values'][sensor] = clip(sensor, machine['values'][sensor])
                elif atype == 'stuck_sensor':
                    machine['values'][sensor] = stuck_sensor(machine)
                    machine['values'][sensor] = clip(sensor, machine['values'][sensor])
                elif atype == 'impossible_value':
                    machine['values'][sensor] = impossible_value(sensor)
            else:
                machine['is_anomaly'][sensor] = 0
                machine['values'][sensor] = clip(sensor, step_normal(machine['values'][sensor], sensor))
                if machine['mode'] == 'ANOMALY':
                    noise_scale = sensor_noise[sensor][1]
                    machine['values'][sensor] += random.uniform(-0.2*noise_scale, 0.2*noise_scale)
                    machine['values'][sensor] = clip(sensor, machine['values'][sensor])
        
        row = {}         
        row['step'] = step+1
        row['machine_id'] = i+1
        
        if machine['mode'] == 'ANOMALY':
            row['any_anomaly'] = 1
        else:
            row['any_anomaly'] = 0
            
        row['target_sensor'] = machine['target_sensor']
        
        for sensor in sensors:
            row[sensor] = machine['values'][sensor]
            row[sensor + '_anomaly'] = machine['is_anomaly'][sensor]
            
        row['anomaly_type'] = machine['anomaly_type']    
        rows.append(row)
        
        if machine['mode'] == 'ANOMALY':
            machine['remaining_duration'] -= 1
            if machine['remaining_duration'] <= 0:
                machine['mode'] = 'NORMAL'
                machine['anomaly_type'] = 'none'
                machine['target_sensor'] = 'none'
                for s in sensors:
                    machine['is_anomaly'][s] = 0
                    
rows_df = pd.DataFrame(rows, index=None)
rows_df.to_csv("sensor_data_raw.csv", index=False)
        
        
                