import random, statistics

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

anomaly_probability = 0.05
anomaly_duration = {
    'spike': (1,3), 
    'drop': (1,3), 
    'drift': (50,100), 
    'oscillation': (30,60), 
    'stuck_sensor': (50,100), 
    'impossible_value': (1,2)
    }

random.seed(random.randint(1, 10000))

def init_machine():
    return {
        'values': {s: random.uniform(*sensor_ranges[s]) for s in sensors},
        'mode': "NORMAL", 
        'anomaly_type': None, 
        'remaining_duration': 0,
        'target_sensor': None, 
        
        'drift_rate': random.uniform(0.005, 0.02),
        'drift_direction': 1,
        'stuck_value': None,
        'osc_center': None,
        'osc_amplitude': None,
        'osc_direction': 1
    }
    
machines = [init_machine() for i in range(3)]

def step_normal(previous_value, sensor):
    lo, hi = sensor_ranges[sensor]
    center = (lo + hi) / 2
    noise = random.uniform(-sensor_noise[sensor][1], sensor_noise[sensor][1])
    return previous_value + 0.1*(center - previous_value) + noise

def spike(sensor):
    lo, hi = sensor_ranges[sensor]
    return hi + random.uniform(0.5*(hi-lo), (hi-lo))
    
def drop(sensor):
    lo, hi = sensor_ranges[sensor]
    return lo - random.uniform(0.5*(hi-lo), (hi-lo))

def drift(previous_value, machine, sensor):
    noise = random.uniform(-(sensor_noise[sensor][1]), sensor_noise[sensor][1])
    return previous_value + machine['drift_direction'] * machine['drift_rate'] + noise

def oscillation(machine, sensor):
    noise = random.uniform(-(sensor_noise[sensor][1]), sensor_noise[sensor][1])
    next_value = machine['osc_center'] + machine['osc_direction'] * machine['osc_amplitude'] + noise
    machine['osc_direction'] *= -1
    return next_value

def stuck_sensor(machine):
    return machine['stuck_value']

def impossible_value(sensor):
    lo, hi = sensor_ranges[sensor]
    if sensor in ['pressure', 'vibration', 'flow_rate']:
        return random.uniform(-10, -1)
    if sensor == 'voltage':
        return 0
    return random.choice[(lo*5, hi*5)]

num_timesteps = 300

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
            
            if machine['anomaly_type'] == 'stuck_sensor':
                machine['stuck_value'] = machine['values'][sensor]
                
            if machine['anomaly_type'] == 'oscillation':
                lo, hi = sensor_ranges[sensor]
                machine['osc_center'] = machine['values'][sensor]
                machine['osc_amplitude'] = 0.2*(hi-lo)
                machine['osc_dir'] = random.choice([-1, 1])
                    
        for sensor in sensors:
            if machine['mode'] == 'ANOMALY' and sensor == machine['target_sensor']:
                type = machine['anomaly_type']
                if type == 'spike':
                    machine['values'][sensor] = spike(sensor)
                elif type == 'drop':
                    machine['values'][sensor] = drop(sensor)
                elif type == 'drift':
                    machine['values'][sensor] = drift(machine['values'][sensor], machine, sensor)
                if type == 'oscillation':
                    machine['values'][sensor] = oscillation(machine, sensor)
                if type == 'stuck_sensor':
                    machine['values'][sensor] = stuck_sensor(machine)
                if type == 'impossible_value':
                    machine['values'][sensor] = impossible_value(sensor)
            else:
                machine['values'][sensor] = step_normal(machine['values'][sensor], sensor)
        
        if machine['mode'] == 'ANOMALY':
            machine['remaining_duration'] -= 1
            if machine['remaining_duration'] <= 0:
                machine['mode'] = 'NORMAL'
                machine['anomaly_type'] = None
                machine['target_sensor'] = None
        
        print(
            f"step={step:03d} | machine={i} | "
            f"T={machine['values']['temperature']:.2f} "
            f"P={machine['values']['pressure']:.2f} "
            f"Vib={machine['values']['vibration']:.2f} | "
            f"mode={machine['mode']:<7} "
            f"type={str(machine['anomaly_type']):<15} "
            f"target={str(machine['target_sensor']):<12} "
            f"remaining={machine['remaining_duration']:03d}"
        )
                