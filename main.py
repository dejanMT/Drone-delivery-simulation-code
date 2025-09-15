#!/usr/bin/env python3
import time
import numpy as np
from math import pi
from dronekit import connect
import rclpy
from markerDetection import MarkerDetection
from navigation import goto_local, goto_gps
from balconyLanding import BalconyLanding
from egress import ascend_and_back
import flightController as fc
import os

vehicle = connect('tcp:127.0.0.1:5763', wait_ready=True)

# For simulation purposes only
CAMERA_MATRIX = np.array([
    [1061.65, 0.0, 640.5],
    [0.0, 1061.65, 360.5],
    [0.0, 0.0, 1.0]
], dtype=np.float32)
DIST_COEFF = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

HORIZONTAL_RES = 600
VERTICAL_RES = 480
HORIZONTAL_FOV = 102 * (pi / 180)
VERTICAL_FOV = 67 * (pi / 180)

TAKEOFF_HEIGHT = 4 #m

latitude = 11    # y
longitude = 16 # x default 16
altitude = TAKEOFF_HEIGHT  # z

controller = fc.FlightController(vehicle=vehicle)
controller.camera_matrix = CAMERA_MATRIX

if __name__ == '__main__':
    marker_detection_node = None 
    try:
        print("Starting mission")
        controller.arm_and_takeoff(TAKEOFF_HEIGHT)
        time.sleep(2)

        # Gazebo gps
        goto_local(vehicle, longitude, latitude, altitude, yaw=0.0)
        # for real-world GPS
        # goto_gps(vehicle, latitude, longitude, altitude)

        rclpy.init()
        marker_detection_node = MarkerDetection()
        marker_detection_node.vehicle = vehicle
        marker_detection_node.controller = controller
        marker_detection_node.CAMERA_MATRIX = CAMERA_MATRIX
        marker_detection_node.DIST_COEFF = DIST_COEFF
        marker_detection_node.HORIZONTAL_RES = HORIZONTAL_RES
        marker_detection_node.VERTICAL_RES = VERTICAL_RES
        marker_detection_node.HORIZONTAL_FOV = HORIZONTAL_FOV
        marker_detection_node.VERTICAL_FOV = VERTICAL_FOV

        while rclpy.ok():
            rclpy.spin_once(marker_detection_node)
            if marker_detection_node.alignment_complete:
                break

        balcony_lander = BalconyLanding(controller, vehicle)
        landing_successful = balcony_lander.execute(controller.latest_image)

        if not landing_successful:
            print("Landing was aborted")
        else:
            print("Landing executed successfully")

        ascend_and_back(vehicle, controller)

    except Exception as e:
        print("An error occurred:", e)
    finally:
        try:
            if marker_detection_node is not None:
                marker_detection_node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        try:
            vehicle.close()
        except Exception:
            pass