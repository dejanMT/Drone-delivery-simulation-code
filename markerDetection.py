#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import cv2.aruco as aruco
import time
import numpy as np
from cv_bridge import CvBridge
from dronekit import VehicleMode
from packaging import version
import os


class MarkerDetection(Node):
    def __init__(self):
        super().__init__('marker_detection')

        self.bridge = CvBridge()

        self.vehicle = None
        self.controller = None

        self.CAMERA_MATRIX = None
        self.DIST_COEFF = None

        self.HORIZONTAL_RES = None
        self.VERTICAL_RES = None
        self.HORIZONTAL_FOV = None
        self.VERTICAL_FOV = None

        self.id_to_find = 18
        self.marker_size = 20.0  # cm
        self.distance_from_marker = 2.0  # m

        try:
            self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        except AttributeError:
            self.aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)

        if version.parse(cv2.__version__) >= version.parse("4.7.0"):
            self.parameters = aruco.DetectorParameters()
            self._use_new_api = True
            self.aruco_detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)
        else:
            try:
                self.parameters = aruco.DetectorParameters_create()
            except AttributeError:
                self.parameters = aruco.DetectorParameters()
            self._use_new_api = False
            self.aruco_detector = None

        # Control gains 
        self.Kp_yaw = 1.0
        self.Kp_x = 0.10
        self.Kp_y = 0.10
        self.Kp_z = 0.10

        self.deadband_px = 20
        self.yaw_alignment_counter = 0

        self.yaw_alignment_complete = False
        self.alignment_complete = False


        self.entry_forward_sign = +1.0
        self.entry_min_forward_speed = 0.07   # m/s
        self.entry_max_forward_speed = 0.15
        self.entry_lateral_speed = 0.12
        self.entry_z_speed = 0.15

        # Drone footprint & safety (m)
        self.drone_width_m = 0.60
        self.clearance_m  = 0.20
        self.base_pad_m   = 0.05
        self.size_pad_scale = 0.20

        # estimate meters/pixel while marker visible
        self.mpp_x = None
        self.mpp_y = None

        self.landing_clear_frames = 0
        self.ready_to_land = False
        self.person_hard_stop = True

        self.yolo_model_path = "yolo11n.onnx"
        self.yolo_img_size = 640
        self.yolo_conf_thr = 0.35
        self.yolo_iou_thr = 0.45
        self.yolo_loaded = False

        self._t_first_frame = None
        self._t_first_marker = None
        self._printed_detect_time = False

        self._t_align_start = None
        self._printed_align_time = False

        self.coco_names = [
            "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
            "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
            "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
            "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
            "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
            "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
            "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
            "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
            "hair drier","toothbrush"
        ]
        self.obstacle_class_names = {
            "chair","couch","potted plant","dining table","bench","tv","bed","refrigerator","microwave",
            "oven","toaster","sink","book","clock","vase","teddy bear","bottle","cup","bowl","backpack",
            "suitcase","handbag","umbrella","bicycle","motorcycle"
        }
        self.person_class_name = "person"

        self.declare_parameter('yolo_model_path', '')
        param_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value.strip()
        if param_path:
            self.yolo_model_path = param_path
        print(f"[YOLO] model='{self.yolo_model_path}'")

        self._load_yolo()

        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.msg_receiver, 10
        )

    def _detect_markers(self, gray_img):
        if self._use_new_api:
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray_img)
        else:
            corners, ids, rejected = aruco.detectMarkers(
                gray_img, dictionary=self.aruco_dict, parameters=self.parameters
            )
        if ids is not None:
            ids = np.array(ids).reshape(-1)
        return corners, ids, rejected

    def _estimate_pose(self, marker_corners):
        if self.CAMERA_MATRIX is None or self.DIST_COEFF is None:
            return None, None
        if hasattr(aruco, 'estimatePoseSingleMarkers'):
            try:
                ret_r, ret_t, _ = aruco.estimatePoseSingleMarkers(
                    marker_corners, self.marker_size,
                    cameraMatrix=self.CAMERA_MATRIX,
                    distCoeffs=self.DIST_COEFF
                )
                rvec = ret_r[0, 0, :]
                tvec = ret_t[0, 0, :]
                return rvec, tvec
            except Exception:
                pass
        s = self.marker_size / 100.0
        half = s / 2.0
        objp = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float32)
        c = marker_corners[0] if marker_corners.ndim == 3 else marker_corners
        imgp = np.array(c, dtype=np.float32)
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, self.CAMERA_MATRIX, self.DIST_COEFF, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            ok, rvec, tvec = cv2.solvePnP(objp, imgp, self.CAMERA_MATRIX, self.DIST_COEFF)
            if not ok:
                return None, None
        return rvec.flatten(), tvec.flatten()

    def _load_yolo(self):
        try:
            path = os.path.abspath(self.yolo_model_path)
            if not os.path.exists(path):
                self.yolo_loaded = False
                return
            self.yolo_net = cv2.dnn.readNetFromONNX(path)
            self.yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.yolo_loaded = True
        except Exception as e:
            print(f"[YOLO] Failed to load ONNX: {e}")
            self.yolo_loaded = False

    def _yolo_infer(self, bgr):
        if not self.yolo_loaded:
            return []
        ih, iw = bgr.shape[:2]
        size = self.yolo_img_size

        # Letterbox
        scale = min(size / iw, size / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        dx, dy = (size - nw) // 2, (size - nh) // 2
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        canvas[dy:dy+nh, dx:dx+nw] = resized

        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        self.yolo_net.setInput(blob)
        out = self.yolo_net.forward()

        # Normalize to (N, C)
        if out.ndim == 3:
            out = np.squeeze(out, 0)
            if out.shape[0] in (84, 85):
                out = out.transpose(1, 0)
        elif out.ndim != 2:
            return []
        preds = out
        if preds.shape[1] < 4:
            return []

        nc = len(self.coco_names)
        C = preds.shape[1]
        if C == 4 + 1 + nc:   
            boxes = preds[:, :4]
            obj = preds[:, 4:5]
            cls_scores = preds[:, 5:]
            cls_ids = np.argmax(cls_scores, axis=1)
            cls_conf = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
            conf = obj[:, 0] * cls_conf
        elif C == 4 + nc:     
            boxes = preds[:, :4]
            cls_scores = preds[:, 4:]
            cls_ids = np.argmax(cls_scores, axis=1)
            conf = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
        else:
            return []

        keep = conf >= self.yolo_conf_thr
        if not np.any(keep): return []
        boxes = boxes[keep]; conf = conf[keep]; cls_ids = cls_ids[keep]

        # cxcywh -> xyxy in letterbox space
        cx, cy, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        x1 = cx - w/2.0; y1 = cy - h/2.0; x2 = cx + w/2.0; y2 = cy + h/2.0
        # Undo letterbox
        x1 = (x1 - dx) / scale; y1 = (y1 - dy) / scale
        x2 = (x2 - dx) / scale; y2 = (y2 - dy) / scale
        x1 = np.clip(x1, 0, iw-1); y1 = np.clip(y1, 0, ih-1)
        x2 = np.clip(x2, 0, iw-1); y2 = np.clip(y2, 0, ih-1)

        rects = np.stack([x1,y1,x2,y2], axis=1).astype(np.float32)
        scores = conf.astype(np.float32)

        idxs = cv2.dnn.NMSBoxes(rects.tolist(), scores.tolist(), self.yolo_conf_thr, self.yolo_iou_thr)
        if len(idxs) == 0: return []
        if isinstance(idxs, (list, tuple)): idxs = [i[0] if isinstance(i,(list,tuple,np.ndarray)) else i for i in idxs]
        else: idxs = np.array(idxs).flatten().tolist()

        dets = []
        for i in idxs:
            cls_id = int(cls_ids[i])
            cname = self.coco_names[cls_id] if 0 <= cls_id < len(self.coco_names) else str(cls_id)
            dets.append((int(rects[i,0]), int(rects[i,1]), int(rects[i,2]), int(rects[i,3]), float(scores[i]), cname))
        return dets

    def _required_free_width_px(self, img_w):
        if self.mpp_x is not None and self.mpp_x > 0:
            return int((self.drone_width_m + 2*self.clearance_m) / self.mpp_x)
        return int(0.25 * img_w)  # conservative fallback

    def _inflate_radius_px(self, img_w):
        if self.mpp_x is not None and self.mpp_x > 0:
            return max(2, int(((self.drone_width_m/2.0) + self.clearance_m) / self.mpp_x))
        return max(2, int(0.08 * img_w))

    def _pov_entry_step(self, frame):
        h, w = frame.shape[:2]
        detections = self._yolo_infer(frame)

        if any(cname == self.person_class_name for *_, cname in detections) and self.person_hard_stop:
            print("Person detected RTL")
            if self.controller is not None:
                self.controller.send_local_ned_velocity(0.0, 0.0, 0.0)
            return

        # Occupancy map
        occ = np.zeros((h, w), dtype=np.uint8)
        if self.mpp_x is not None and self.mpp_x > 0:
            base_px = max(2, int(self.base_pad_m / self.mpp_x))
        else:
            base_px = max(2, int(0.02 * w))

        for (x1, y1, x2, y2, conf, cname) in detections:
            if (cname in self.obstacle_class_names) or (cname == self.person_class_name):
                bw = x2 - x1
                bh = y2 - y1
                pad = base_px + int(self.size_pad_scale * max(bw, bh))
                xx1 = max(0, x1 - pad); yy1 = max(0, y1 - pad)
                xx2 = min(w - 1, x2 + pad); yy2 = min(h - 1, y2 + pad)
                occ[yy1:yy2+1, xx1:xx2+1] = 255

        radius_px = self._inflate_radius_px(w)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius_px+1, 2*radius_px+1))
        occ_infl = cv2.dilate(occ, kernel, iterations=1)

        # Choose free corridor in middle band
        band_w = int(0.40 * w)
        band_x1 = (w - band_w) // 2
        band_x2 = band_x1 + band_w
        corridor = occ_infl[:, band_x1:band_x2]

        col_free = (corridor == 0).sum(axis=0)
        min_col_free = int(0.60 * h)
        req_w_px = self._required_free_width_px(w)

        widest = (0, -1, 0); start = None
        for i in range(col_free.shape[0]):
            if col_free[i] >= min_col_free:
                if start is None: start = i
            else:
                if start is not None:
                    width = i - start
                    if width >= req_w_px and width > widest[2]:
                        widest = (start, i-1, width)
                    start = None
        if start is not None:
            width = col_free.shape[0] - start
            if width >= req_w_px and width > widest[2]:
                widest = (start, col_free.shape[0]-1, width)

        # Default crawl forward (POV)
        vx = self.entry_forward_sign * self.entry_min_forward_speed
        vy = 0.0
        vz = 0.0

        if widest[2] > 0:
            free_mid_col = band_x1 + (widest[0] + widest[1]) // 2
            img_mid = w // 2
            px_err = free_mid_col - img_mid
            if abs(px_err) > 8:
                vy = np.sign(px_err) * self.entry_lateral_speed
            front_rows = corridor[:int(0.30*h), :]
            if (front_rows == 0).mean() > 0.90:
                vx = self.entry_forward_sign * min(self.entry_max_forward_speed, abs(vx) + 0.05)
        else:
            print(f"No corridor ≥ {req_w_px}px — re-centering.")
            vx = 0.0
            left_free = (occ_infl[:, :w//2] == 0).mean()
            right_free = (occ_infl[:, w//2:] == 0).mean()
            if right_free > left_free + 0.05: vy = +self.entry_lateral_speed
            elif left_free > right_free + 0.05: vy = -self.entry_lateral_speed
            else: vy = 0.0


        lh_y1 = int(0.70 * h)
        lw = max(req_w_px, int(0.35 * w))
        lw_x1 = max(0, (w - lw) // 2)
        lw_x2 = min(w, lw_x1 + lw)
        landing_patch = occ_infl[lh_y1:h, lw_x1:lw_x2]
        free_ratio = (landing_patch == 0).mean()

        if free_ratio > 0.92:
            self.landing_clear_frames += 1
        else:
            self.landing_clear_frames = 0

        if self.landing_clear_frames >= 5:
            self.ready_to_land = True

        if self.ready_to_land:
            print("Clear landing patch. Descending…")
            vz = +self.entry_z_speed               # positive down
            vx = self.entry_forward_sign * min(abs(vx), 0.05)  

        if self.controller is not None:
            self.controller.send_local_ned_velocity(vx, vy, vz)

        print(f"[POV] vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}, req_w_px={req_w_px}, free_ratio={free_ratio:.2f}")


    def msg_receiver(self, message):
        # Convert ROS Image -> BGR
        np_data = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        if self._t_first_frame is None:
            self._t_first_frame = time.monotonic()

        if self.controller is not None:
            self.controller.latest_image = np_data

        if self.HORIZONTAL_RES is None or self.VERTICAL_RES is None:
            h, w = np_data.shape[:2]
            self.HORIZONTAL_RES = float(w)
            self.VERTICAL_RES = float(h)

        gray_img = cv2.cvtColor(np_data, cv2.COLOR_BGR2GRAY)

        if self.alignment_complete:
            self._pov_entry_step(np_data)
            time.sleep(0.08)
            return
        corners, ids, _ = self._detect_markers(gray_img)

        if ids is not None and self.id_to_find in ids:
            # Timer for first marker detection
            if self._t_first_marker is None:
                self._t_first_marker = time.monotonic()
                if self._t_first_frame is not None and not self._printed_detect_time:
                    dt = self._t_first_marker - self._t_first_frame
                    print(f"Marker detected in {dt:.2f}s from first frame.")
                    self._printed_detect_time = True
                self._t_align_start = self._t_first_marker

                if not self.yaw_alignment_complete and self._t_align_start is None:
                     self._t_align_start = time.monotonic()

            index = list(ids).index(self.id_to_find)

            rvec, tvec = self._estimate_pose(corners[index])
            if rvec is None or tvec is None:
                print("Pose estimation unavailable (missing intrinsics or API). Hovering.")
                if self.controller is not None:
                    self.controller.send_local_ned_velocity(0.0, 0.0, 0.0)
                time.sleep(0.1)
                return

            x_avg = np.mean(corners[index][0][:, 0])
            center_x = self.HORIZONTAL_RES / 2.0
            x_error = x_avg - center_x

            # Yaw alignment
            if not self.yaw_alignment_complete:
                x_avg = np.mean(corners[index][0][:, 0])
                center_x = self.HORIZONTAL_RES / 2.0
                x_error = x_avg - center_x

                yaw_aligned = abs(x_error) <= self.deadband_px

                if yaw_aligned:
                    self.yaw_alignment_counter += 1
                else:
                    self.yaw_alignment_counter = 0

                if self.yaw_alignment_counter >= 1:
                    print("Yaw alignment complete")
                    if self.controller is not None:
                        self.controller.send_yaw_rate(0.0)        
                        self.controller.send_local_ned_velocity(0,0,0)   
                    self.yaw_alignment_complete = True
                    return
                else:
                    yaw_rate = -self.Kp_yaw * (x_error / center_x)
                    yaw_rate = max(min(yaw_rate, 0.4), -0.4)
                    if self.controller is not None:
                        self.controller.send_yaw_rate(yaw_rate) 
                    time.sleep(0.1)
                    return
                
            marker_distance = float(tvec[2])

            camera_to_center_offset = 0.20
            corrected_distance = marker_distance + camera_to_center_offset
            if self.controller is not None:
                self.controller.marker_distance = corrected_distance

            if self.CAMERA_MATRIX is not None:
                fx = float(self.CAMERA_MATRIX[0][0])
                fy = float(self.CAMERA_MATRIX[1][1])
                mpp_x_now = corrected_distance / max(fx, 1e-6)
                mpp_y_now = corrected_distance / max(fy, 1e-6)
                if self.mpp_x is None:
                    self.mpp_x, self.mpp_y = mpp_x_now, mpp_y_now
                else:
                    self.mpp_x = 0.8*self.mpp_x + 0.2*mpp_x_now
                    self.mpp_y = 0.8*self.mpp_y + 0.2*mpp_y_now
            else:
                fx = self.HORIZONTAL_RES
                fy = self.VERTICAL_RES

            x_px_offset = x_avg - center_x
            target_y_position = self.VERTICAL_RES * 0.9
            y_avg = np.mean(corners[index][0][:, 1])
            y_px_offset = y_avg - target_y_position

            lateral_error = (x_px_offset / fx) * marker_distance
            vertical_error = (y_px_offset / fy) * marker_distance
            error_forward = corrected_distance - self.distance_from_marker

            forward_aligned = abs(error_forward) <= 0.50
            lateral_aligned = abs(lateral_error) <= 0.45
            vertical_aligned = abs(vertical_error) <= 0.15

            vx = self.Kp_z * error_forward if not forward_aligned else 0.0
            vy = self.Kp_x * lateral_error if not lateral_aligned else 0.0
            vz = self.Kp_y * vertical_error if not vertical_aligned else 0.0

            vx = max(min(vx, 0.2), -0.2)
            vy = max(min(vy, 0.2), -0.2)
            vz = max(min(vz, 0.2), -0.2)

            if lateral_aligned and vertical_aligned:
                if self.controller is not None:
                    self.controller.send_local_ned_velocity(0.0, 0.0, -0.2)  # ascend
                time.sleep(0.3)
                if self.controller is not None:
                    self.controller.send_local_ned_velocity(0.0, 0.0, 0.0)

                marker_corners = corners[index]
                c = marker_corners[0]  # (4,2)
                marker_px_width = np.linalg.norm(c[0] - c[1])
                marker_px_height = np.linalg.norm(c[0] - c[3])

                marker_real_size = self.marker_size / 100.0  # meters
                meters_per_pixel_x = marker_real_size / marker_px_width if marker_px_width > 0 else 0.0
                meters_per_pixel_y = marker_real_size / marker_px_height if marker_px_height > 0 else 0.0

                estimated_balcony_width = self.HORIZONTAL_RES * meters_per_pixel_x
                estimated_balcony_height = self.VERTICAL_RES * meters_per_pixel_y

                print(f"Balcony Width: {estimated_balcony_width:.2f} m")
                print(f"Balcony Height: {estimated_balcony_height:.2f} m")

                drone_width = 0.6
                drone_height = 0.2

                if estimated_balcony_width < drone_width + 0.2 or estimated_balcony_height < drone_height + 0.2:
                    print("Balcony too small. Aborting entry.")
                    if self.controller is not None:
                        self.controller.send_local_ned_velocity(0.0, 0.0, 0.0)
                    return
                # timer - to timer the alignment
                if self._t_align_start is not None and not self._printed_align_time:
                    dt_align = time.monotonic() - self._t_align_start
                    print(f"Alignment completed in {dt_align:.2f}s from first detection.")
                    self._printed_align_time = True

                self.alignment_complete = True  
                return

            if self.controller is not None:
                self.controller.send_local_ned_velocity(vx, vy, vz)

        time.sleep(0.1)

    def run(self):
        rclpy.spin(self)
        self.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    rclpy.init()
    node = MarkerDetection()
    node.run()
