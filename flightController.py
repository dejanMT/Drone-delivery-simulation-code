#!/usr/bin/env python
from __future__ import print_function
import time
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil

class FlightController(object):
    def __init__(self, vehicle=None, connection_string='tcp:127.0.0.1:5763'):
            if vehicle is None:
                print("Connecting to vehicle on: {}".format(connection_string))
                self.vehicle = connect(connection_string, wait_ready=True)
            else:
                self.vehicle = vehicle


            self.vehicle.parameters['PLND_ENABLED'] = 0
            self.vehicle.parameters['PLND_TYPE'] = 0
            self.vehicle.parameters['PLND_EST_TYPE'] = 0
            self.vehicle.parameters['LAND_SPEED'] = 30

            self.camera_matrix = None



    def arm_and_takeoff(self, target_altitude):
        print("Waiting for vehicle to become armable")
        while not self.vehicle.is_armable:
            print("Waiting for vehicle to be armable")
            time.sleep(1)

        print("Setting mode to GUIDED")
        self.vehicle.mode = VehicleMode("GUIDED")
        
        while self.vehicle.mode.name != "GUIDED":
            print("Waiting for GUIDED mode")
            time.sleep(1)
        
        print("Arming motors")
        self.vehicle.armed = True
        while not self.vehicle.armed:
            print("Waiting for arming")
            time.sleep(1)

        print("Taking off")
        self.vehicle.simple_takeoff(target_altitude)


        while True:
            alt = self.vehicle.location.global_relative_frame.alt
            if alt >= target_altitude * 0.95:
               break  # Drone reached location
            time.sleep(1)



    def send_local_ned_velocity(self, vx, vy, vz):

        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0, 0, 0,  # time_boot_ms, target system, target component
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111000111,  # type_mask 
            0, 0, 0,  # x, y, z positions 
            vx, vy, vz,  # velocity components in m/s
            0, 0, 0,  # acceleration (
            0, 0)     # yaw, yaw_rate 
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()

    def land(self):

        self.vehicle.mode = VehicleMode("LAND")
        while self.vehicle.armed:
            print("Waiting for landing to complete")
            time.sleep(1)
        print("Landed.")

    def close(self):
        self.vehicle.close()
        print("Vehicle connection closed")
        
    def send_yaw_rate(self, yaw_rate):

        msg = self.vehicle.message_factory.set_attitude_target_encode(
            0,  # time_boot_ms
            0, 0,  # target system, target component
            0b00000111,  # Ignore roll, pitch, but control yaw rate
            [1, 0, 0, 0],  # No roll/pitch control 
            0, 0, yaw_rate,  # Roll, Pitch, Yaw rate
            0.5  # Add thrust to maintain hover
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()

        
    
    def send_body_velocity(vehicle, yaw_rate=0):

        msg = vehicle.message_factory.set_attitude_target_encode(
            0, 0, 0,
            0b00000111,  # Ignore roll and pitch, control yaw rate only
            [1, 0, 0, 0],  # Neutral quaternion (no roll/pitch adjustment)
            0, 0, yaw_rate,
            0
        )
        vehicle.send_mavlink(msg)
        vehicle.flush()

    def get_camera_frame(self):
        if hasattr(self, 'latest_image'):
            return self.latest_image
        return None
