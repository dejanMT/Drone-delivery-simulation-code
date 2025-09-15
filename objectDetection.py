from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n.pt")  

PERSON_CLASS_ID = 0
FURNITURE_CLASSES = {
    56: "chair",
    57: "couch",
    61: "dining table",
}

def detect_furniture(image: np.ndarray):
    if image is None:
        return []

    results = model(image, verbose=False)
    obstacles = []

    for r in results:
        if r.boxes is not None:
            boxes = r.boxes
            for i in range(len(boxes.cls)):
                cls_id = int(boxes.cls[i].item())
                if cls_id in FURNITURE_CLASSES:
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    box_height = y2 - y1
                    box_width = x2 - x1
                    aspect_ratio = box_width / (box_height + 1e-5)

                    image_mid_y = image.shape[0] // 2
                    if box_height < 80 and aspect_ratio > 2.5 and y2 < image_mid_y:
                        continue  


                    x_center = (x1 + x2) / 2
                    obstacles.append({
                        'class_id': cls_id,
                        'label': FURNITURE_CLASSES[cls_id],
                        'x_center': float(x_center),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })

    return obstacles



def detect_clear_floor_zones(image: np.ndarray) -> list:
  
    results = model(image, verbose=False)
    height, width = image.shape[:2]

    mask = np.zeros((height, width), dtype=np.uint8)

    for r in results:
        if r.boxes is not None:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)  
    clear_mask = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(clear_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    zones = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:  
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                zones.append({'x_center': cx, 'y_center': cy})
    return zones

