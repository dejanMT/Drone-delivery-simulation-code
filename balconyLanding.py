import time
import cv2
import numpy as np
import sys
import os
from dronekit import VehicleMode

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from objectDetection import detect_furniture, detect_clear_floor_zones

class BalconyLanding:
    def __init__(self, controller, vehicle):
        self.controller = controller
        self.vehicle = vehicle

    def execute(self, image_frame):

        print("Landing on balcony")

        marker_distance = getattr(self.controller, 'marker_distance', 2.0)
        fy = self.controller.camera_matrix[1][1] if self.controller.camera_matrix is not None else 1000
        meters_per_pixel = marker_distance / fy
        drone_box_m = 0.6

        forward_distance = 6.0
        step_size = 0.2


        for _ in range(int(forward_distance / step_size)):
            self.controller.send_local_ned_velocity(step_size, 0.0, 0.0)
            time.sleep(0.5)
        self.controller.send_local_ned_velocity(0.0, 0.0, 0.0)
        time.sleep(1.0)

        image_center_x = image_frame.shape[1] // 2
        height, width = image_frame.shape[:2]

        max_attempts = 5
        attempts = 0
        landed = False

        while attempts < max_attempts:
            image_frame = self.controller.latest_image
            furniture = detect_furniture(image_frame)
            furniture_boxes = [f['bbox'] for f in furniture]
            clear_zones = detect_clear_floor_zones(image_frame)

            if clear_zones:
                best_zone = min(clear_zones, key=lambda z: abs(z['x_center'] - image_center_x))
                dy = (best_zone['x_center'] - image_center_x) / fy * marker_distance

                for _ in range(5):
                    self.controller.send_local_ned_velocity(0.0, np.clip(dy, -0.3, 0.3), 0.0)
                    time.sleep(0.5)

                self.controller.send_local_ned_velocity(0.0, 0.0, 0.0)
                time.sleep(0.5)

                print("Landing")
                self.vehicle.mode = VehicleMode("LAND")
                landed = True
                break



            self.controller.send_local_ned_velocity(0.0, -0.3, 0.0)
            time.sleep(0.4)
            self.controller.send_local_ned_velocity(0.0, 0.0, 0.0)
            time.sleep(0.5)
            attempts += 1

        if not landed:
            self.vehicle.mode = VehicleMode("RTL")
            return False
        
        return True