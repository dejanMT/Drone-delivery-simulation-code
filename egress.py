import time
from dronekit import VehicleMode

def ascend_and_back(vehicle, controller):
    time.sleep(18)
    vehicle.mode = VehicleMode("GUIDED")
    
    try:
        start_alt = float(vehicle.location.global_relative_frame.alt or 0.0)
    except Exception:
        start_alt = 0.0
    target_alt = start_alt + 1.2

    while True:
        try:
            curr_alt = float(vehicle.location.global_relative_frame.alt or 0.0)
        except Exception:
            curr_alt = start_alt
        if curr_alt >= target_alt - 0.05:
            break
        controller.send_local_ned_velocity(0.0, 0.0, -0.5)
        time.sleep(2)
    controller.send_local_ned_velocity(0.0, 0.0, 0.0)

    moved = 0.0
    step_speed = -0.5  
    dt = 0.2
    while moved < 4.0:
        controller.send_local_ned_velocity(step_speed, 0.0, 0.0)
        time.sleep(dt)
        moved += abs(step_speed) * dt
    controller.send_local_ned_velocity(0.0, 0.0, 0.0)

    vehicle.mode = VehicleMode("RTL")
