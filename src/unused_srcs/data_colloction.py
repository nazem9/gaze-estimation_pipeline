import cv2
import os
import random
from ultralytics import YOLO
from datetime import datetime

model = YOLO('yolo11l.pt').cuda()
cap = cv2.VideoCapture(0)

output_dir = 'gaze_dataset_yolo'
os.makedirs(output_dir, exist_ok=True)
image_dir = os.path.join(output_dir, 'images')
os.makedirs(image_dir, exist_ok=True)
label_dir = os.path.join(output_dir, 'labels')
os.makedirs(label_dir, exist_ok=True)

# Create dataset.yaml file for YOLO training
with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
    f.write("train: ./images\n")
    f.write("val: ./images\n\n")
    f.write("nc: 2\n")  # 2 classes: smartphone and gaze point
    f.write("names: ['smartphone', 'gaze_point']\n")

crosshair_color = (0, 255, 0)
crosshair_radius = 5
gaze_x, gaze_y = None, None

cv2.namedWindow('Gaze & Smartphone Detection', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Gaze & Smartphone Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Flip the frame horizontally to correct mirroring
    frame = cv2.flip(frame, 1)
    
    original_frame = frame.copy()
    height, width = frame.shape[:2]

    if gaze_x is None or gaze_y is None:
        gaze_x = random.randint(0, width - 1)
        gaze_y = random.randint(0, height - 1)

    frame = cv2.circle(frame, (gaze_x, gaze_y), crosshair_radius, crosshair_color, -1)

    detections = model.predict(original_frame, classes=[67], stream=True, verbose=False)

    boxes = []
    for detection in detections:
        if detection.boxes.cpu():
            for box in detection.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('Gaze & Smartphone Detection', frame)
    key = cv2.waitKey(5) & 0xFF

    if key == ord('c'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f'frame_{timestamp}'
        
        # Save image
        image_path = os.path.join(image_dir, f'{filename}.jpg')
        cv2.imwrite(image_path, original_frame)
        
        # Create YOLO format label file
        label_path = os.path.join(label_dir, f'{filename}.txt')
        with open(label_path, 'w') as f:
            # Write gaze point (class 1)
            # Convert coordinates to normalized format (0-1)
            norm_gaze_x = gaze_x / width
            norm_gaze_y = gaze_y / height
            # For gaze point, use a small width and height (e.g., 0.01)
            gaze_width = gaze_height = 0.01
            f.write(f"1 {norm_gaze_x:.6f} {norm_gaze_y:.6f} {gaze_width:.6f} {gaze_height:.6f}\n")
            
            # Write smartphone boxes (class 0)
            for box in boxes:
                x1, y1, x2, y2 = box
                # Convert to YOLO format: class x_center y_center width height
                x_center = (x1 + x2) / (2 * width)
                y_center = (y1 + y2) / (2 * height)
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                f.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

        print(f"Saved {filename} with annotations")
        gaze_x = None
        gaze_y = None
        cv2.waitKey(300)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()