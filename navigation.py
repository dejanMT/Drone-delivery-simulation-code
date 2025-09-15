from dronekit import LocationGlobalRelative
import time
from pymavlink import mavutil

def goto_gps(vehicle, latitude, longitude, altitude):
    """Fly to a specified GPS coordinate."""
    target_location = LocationGlobalRelative(latitude, longitude, altitude)
    vehicle.simple_goto(target_location)
    time.sleep(10)

def goto_local(vehicle, x, y, z, yaw=None):
    """
    Move the drone to a specific local coordinate.
    This function uses MAVLink's SET_POSITION_TARGET_LOCAL_NED to move
    the drone in the local frame.
    """

    type_mask = 0b0000111111111000  # position only

    if yaw is not None:
        type_mask = 0b0000111111110000  # include yaw

    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        x, y, -z,
        0, 0, 0,
        0, 0, 0,
        yaw if yaw is not None else 0,
        0
    )

    vehicle.send_mavlink(msg)
    vehicle.flush()


    while True:
        current_position = vehicle.location.local_frame
        dist_x = abs(current_position.north - x)
        dist_y = abs(current_position.east - y)
        dist_z = abs(current_position.down + z)  

        if dist_x < 0.2 and dist_y < 0.2 and dist_z < 0.2:
            break

        time.sleep(1)
