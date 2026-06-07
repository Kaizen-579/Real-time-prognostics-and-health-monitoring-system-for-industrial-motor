import socket
import struct
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
from collections import deque

#CONFIG

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

model = joblib.load("motor_fault_model.pkl")
health_history = deque(maxlen=200)

#history buffers
temp_hist = deque(maxlen=50)
curr_hist = deque(maxlen=50)
speed_hist = deque(maxlen=50)
torque_hist = deque(maxlen=50)

#STABILITY BUFFERS
efficiency_hist = deque(maxlen=20)
fault_memory = deque(maxlen=20)
conf_memory = deque(maxlen=10)

#FAULT CONTROL
current_fault = "NONE"
fault_hold_counter = 0
FAULT_HOLD_TIME = 12

#BASELINE FREEZE
freeze_baseline = False

FAILURE_HEALTH = 40
MAX_RUL_DAYS = 365
MAX_SERVICE_DAYS = 180

damage = 0.0
last_time = time.time()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print("PHM Started...\n")

def compute_health_continuous(value, baseline, scale, reverse=False):
    if not reverse:
        x = (value - baseline) / (scale + 1e-6)
    else:
        x = (baseline - value) / (scale + 1e-6)
    health = 100 / (1 + np.exp(2 * x))
    return np.clip(health, 5, 100)

#DASHBOARD

plt.ion()
fig = plt.figure(figsize=(10,6))
fig.patch.set_facecolor('#0b0f1a')
ax_text = fig.add_subplot(111)
ax_text.axis('off')

last_update = time.time()

while True:

    data, addr = sock.recvfrom(1024)
    if len(data) < 40:
        continue

    voltage, current, speed, torque, temperature = struct.unpack('<5d', data[:40])

    #STORE HISTORY

    if not freeze_baseline:
        temp_hist.append(temperature)
        curr_hist.append(current)
        speed_hist.append(speed)
        torque_hist.append(torque)

    #BASELINES

    temp_base = np.mean(temp_hist) if len(temp_hist) > 10 else 70
    curr_base = np.mean(curr_hist) if len(curr_hist) > 10 else 8
    speed_base = np.mean(speed_hist) if len(speed_hist) > 10 else 1500
    torque_base = np.mean(torque_hist) if len(torque_hist) > 10 else 10

    #HEALTH

    temp_h = compute_health_continuous(temperature, temp_base, 30)
    curr_h = compute_health_continuous(current, curr_base, 8)
    speed_h = compute_health_continuous(speed, speed_base, 300, reverse=True)
    torque_h = compute_health_continuous(torque, torque_base, 18)

    overall_health = (
        0.35 * temp_h +
        0.25 * curr_h +
        0.20 * speed_h +
        0.20 * torque_h
    )

    #HEALTH BIAS (push to 90–100 when healthy)
    overall_health = 0.6 * overall_health + 40

    temp_factor = max(0, (temperature - temp_base) / 25)
    curr_factor = max(0, (current - curr_base) / 7)

    overall_health -= (temp_factor * curr_factor) * 10
    overall_health = np.clip(overall_health, 20, 100)

    #SMOOTH

    health_history.append(overall_health)
    smooth_health = np.mean(health_history)

    #ML

    try:
        features = pd.DataFrame([{
            "current": current,
            "speed": speed,
            "temperature": temperature,
            "torque": torque,
            "current_speed": current / (speed + 1e-6),
            "torque_current": torque / (current + 1e-6)
        }])
        prediction = model.predict(features.values)[0]
        ml_status = "ANOMALY" if prediction == -1 else "NORMAL"
    except:
        ml_status = "ERROR"

    #FAULT SCORES

    speed_factor = abs(speed - speed_base) / (speed_base + 1e-6)
    torque_factor = max(0, (torque - torque_base) / 15)

    overheat_score = temp_factor * 1.2 + curr_factor * 0.3  # PRIORITY BOOST
    overload_score = curr_factor * 0.7 + torque_factor * 0.3
    drag_score = speed_factor * 0.6 + curr_factor * 0.4
    misalign_score = torque_factor * 0.6 + speed_factor * 0.4

    scores = {
        "OVERHEATING": overheat_score,
        "OVERLOAD": overload_score,
        "MECHANICAL DRAG": drag_score,
        "MISALIGNMENT": misalign_score
    }

    sorted_faults = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    best_fault, best_conf = sorted_faults[0]
    second_fault, second_conf = sorted_faults[1]

    conf_memory.append(best_conf)
    smooth_conf = np.mean(conf_memory)

    threshold = 0.3

    #FAULT LOGIC (DOMINANCE LOCK)

    if smooth_conf > threshold:

        freeze_baseline = True  # 🔥 freeze learning

        if current_fault == "NONE":
            current_fault = best_fault
            fault_hold_counter = FAULT_HOLD_TIME

        elif best_fault != current_fault:
            if best_conf > (smooth_conf + 0.2):
                current_fault = best_fault
                fault_hold_counter = FAULT_HOLD_TIME

        else:
            fault_hold_counter = FAULT_HOLD_TIME

    else:
        freeze_baseline = False

        if fault_hold_counter <= 0:
            current_fault = "NONE"
        else:
            fault_hold_counter -= 1

    fault = current_fault
    fault_conf = smooth_conf

    #EFFICIENCY

    angular_speed = speed * 2 * np.pi / 60
    mechanical_power = torque * angular_speed
    electrical_power = voltage * current + 1e-6

    raw_eff = mechanical_power / electrical_power
    raw_eff = max(0.0, min(raw_eff, 1.0))

    efficiency_hist.append(raw_eff)
    efficiency = np.mean(efficiency_hist)

    #COUPLING FIX
    if smooth_health < 70:
        efficiency *= 0.75

    #RUL

    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time

    stress = (
        0.4 * temp_factor +
        0.25 * curr_factor +
        0.2 * torque_factor +
        0.15 * speed_factor
    )

    base_life_seconds = 180 * 24 * 3600
    base_rate = 1 / base_life_seconds

    damage_rate = base_rate * (1 + 3 * (stress ** 2))

    damage += damage_rate * dt
    damage = np.clip(damage, 0, 1)

    remaining_seconds = (1 - damage) / (damage_rate + 1e-9)
    failure_days = remaining_seconds / (3600 * 24)

    failure_days *= (smooth_health / 100.0)
    service_days = failure_days * 0.6

    failure_days = np.clip(failure_days, 1, MAX_RUL_DAYS)
    service_days = np.clip(service_days, 1, MAX_SERVICE_DAYS)

    #REFRESH

    if time.time() - last_update < 0.5:
        continue

    last_update = time.time()

    ax_text.clear()
    ax_text.axis('off')

    health_color = "lime" if smooth_health > 90 else "yellow" if smooth_health > 70 else "red"

    dashboard_text = f"""
_________________________
     MOTOR PANEL
_________________________

Voltage     : {voltage:.2f} V
Current     : {current:.2f} A
Speed       : {speed:.2f} RPM
Torque      : {torque:.2f} Nm
Temperature : {temperature:.2f} °C

Health (Smooth) : {smooth_health:.2f} %

ML Status : {ml_status}
Fault     : {fault} ({fault_conf*100:.0f}%)
Suspect 2 : {second_fault} ({second_conf*100:.0f}%)

Efficiency : {efficiency*100:.2f} %

Service Due : {service_days:.0f} days
Failure In  : {failure_days:.0f} days
"""

    ax_text.text(
        0.02, 0.95,
        dashboard_text,
        fontsize=12,
        color=health_color,
        verticalalignment='top',
        family='monospace'
    )

    plt.pause(0.01)
