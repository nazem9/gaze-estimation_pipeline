import mediapipe as mp
import cv2
import time
import datetime
import os
import numpy as np
import math
from landmarks import *
from face_model import *
from AffineTransformer import AffineTransformer
from EyeballDetector import EyeballDetector
from ultralytics import YOLO

model_path = "./face_landmarker.task"
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class GazeRegion:
    def __init__(self):
        # Define angles in degrees
        self.center_horizontal = 30  # Acceptable horizontal angle for center viewing
        self.center_vertical = 20    # Acceptable vertical angle for center viewing
        self.peripheral_horizontal = 45  # Max acceptable horizontal angle
        self.peripheral_vertical = 30    # Max acceptable vertical angle
        
        # Define duration thresholds (in frames)
        self.peripheral_duration_threshold = 20  # How long gaze can stay in peripheral vision
        self.outside_duration_threshold = 10      # How long before triggering warning for outside viewing

class GazeProcessor:
    def __init__(self, camera_idx=0, callback=None, visualization_options=None):
        self.camera_idx = camera_idx
        self.callback = callback
        self.vis_options = visualization_options
        self.left_detector = EyeballDetector(DEFAULT_LEFT_EYE_CENTER_MODEL)
        self.right_detector = EyeballDetector(DEFAULT_RIGHT_EYE_CENTER_MODEL)
        self.options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO
        )
        self.smartphone_detector = YOLO('yolo11l.pt')
        self.phone_detection_frames = 0
        self.phone_detection_threshold = 5
        self.gaze_history_left = []
        self.gaze_history_right = []
        self.history_length = 15
        self.gaze_region = GazeRegion()

    def calculate_gaze_angle(self, vector):
        norm = np.linalg.norm(vector)
        if norm < 1e-6:
            return 0, 0
        normalized = vector / norm
        horizontal_angle = np.degrees(np.arcsin(normalized[0]))
        vertical_angle = np.degrees(np.arcsin(normalized[1]))
        return horizontal_angle, vertical_angle


    def check_gaze_zone(self, h_angle, v_angle):
        if (abs(h_angle) <= self.gaze_region.center_horizontal and 
            abs(v_angle) <= self.gaze_region.center_vertical):
            return "CENTER"
        elif (abs(h_angle) <= self.gaze_region.peripheral_horizontal and 
              abs(v_angle) <= self.gaze_region.peripheral_vertical):
            return "PERIPHERAL"
        else:
            return "OUTSIDE"

    def analyze_gaze_pattern(self, history):
        if not history:
            return False
        
        consecutive_peripheral = 0
        consecutive_outside = 0
        
        for zone in reversed(history):
            if zone == "PERIPHERAL":
                consecutive_peripheral += 1
            elif zone == "OUTSIDE":
                consecutive_outside += 1
            else:
                consecutive_peripheral = 0
                consecutive_outside = 0
                
        return (consecutive_peripheral >= self.gaze_region.peripheral_duration_threshold or 
                consecutive_outside >= self.gaze_region.outside_duration_threshold)

    def is_gaze_out_of_bounds(self, vector, is_left_eye=True):
        horizontal_angle, vertical_angle = self.calculate_gaze_angle(vector)
        gaze_zone = self.check_gaze_zone(horizontal_angle, vertical_angle)
        
        history = self.gaze_history_left if is_left_eye else self.gaze_history_right
        history.append(gaze_zone)
        
        if len(history) > self.history_length:
            history = history[-self.history_length:]
        
        if is_left_eye:
            self.gaze_history_left = history
        else:
            self.gaze_history_right = history
            
        return self.analyze_gaze_pattern(history)

    def draw_gaze_regions(self, frame, head_center):
        h, w = frame.shape[:2]
        center_x, center_y = head_center

        # Draw center zone
        center_w = int(w * math.tan(math.radians(self.gaze_region.center_horizontal)))
        center_h = int(h * math.tan(math.radians(self.gaze_region.center_vertical)))
        cv2.rectangle(frame,
                     (center_x - center_w, center_y - center_h),
                     (center_x + center_w, center_y + center_h),
                     self.vis_options.color, self.vis_options.box_thickness)
        
        # Draw peripheral zone
        periph_w = int(w * math.tan(math.radians(self.gaze_region.peripheral_horizontal)))
        periph_h = int(h * math.tan(math.radians(self.gaze_region.peripheral_vertical)))
        cv2.rectangle(frame,
                     (center_x - periph_w, center_y - periph_h),
                     (center_x + periph_w, center_y + periph_h),
                     self.vis_options.peripheral_color, self.vis_options.box_thickness)

    def draw_gaze_vector(self, frame, start_point, gaze_vector, is_out_of_bounds):
        scale = 100
        end_point = (
            int(start_point[0] + gaze_vector[0] * scale),
            int(start_point[1] + gaze_vector[1] * scale)
        )
        color = self.vis_options.warning_color if is_out_of_bounds else self.vis_options.color
        cv2.arrowedLine(frame, start_point, end_point, color, self.vis_options.line_thickness)

    def draw_gaze_status(self, frame, left_status, right_status):
        h = frame.shape[0]
        cv2.putText(frame, f"Left Eye: {left_status}", 
                    (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    self.vis_options.text_scale, self.vis_options.color, 2)
        cv2.putText(frame, f"Right Eye: {right_status}", 
                    (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    self.vis_options.text_scale, self.vis_options.color, 2)

    def log_cheating_event(self, reason, frame):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = "cheating_logs"
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, f"{timestamp}_{reason.replace(' ', '_')}.jpg")
        cv2.imwrite(filename, frame)
        with open(os.path.join(folder, "log.txt"), "a") as f:
            f.write(f"{timestamp}: {reason}\n")

    async def start(self):
        with FaceLandmarker.create_from_options(self.options) as landmarker:
            cap = cv2.VideoCapture(self.camera_idx)
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    continue
                frame = cv2.flip(frame, 1)
                # Phone detection
                detections = self.smartphone_detector.predict(frame, classes=[67], stream=True, verbose=False)
                phone_detected = False
                for detection in detections:
                    if detection.boxes:
                        phone_detected = True
                        frame = detection.plot()
                if phone_detected:
                    self.phone_detection_frames += 1
                else:
                    self.phone_detection_frames = 0
                if self.phone_detection_frames >= self.phone_detection_threshold:
                    self.log_cheating_event("Phone detected", frame)
                    self.phone_detection_frames = 0

                # Face landmark detection
                timestamp_ms = int(time.time() * 1000)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                face_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if face_landmarker_result.face_landmarks:
                    lms_s = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarker_result.face_landmarks[0]])
                    lms_2 = (lms_s[:, :2] * [frame.shape[1], frame.shape[0]]).round().astype(int)
                    
                    # Calculate head center for visualization
                    nose_tip = lms_2[NOSE_TIP]
                    head_center = (nose_tip[0], nose_tip[1])

                    if self.vis_options:
                        self.draw_gaze_regions(frame, head_center)

                    # Setup affine transformer
                    mp_hor_pts = [lms_s[i] for i in OUTER_HEAD_POINTS]
                    mp_ver_pts = [lms_s[i] for i in [NOSE_BRIDGE, NOSE_TIP]]
                    model_hor_pts = OUTER_HEAD_POINTS_MODEL
                    model_ver_pts = [NOSE_BRIDGE_MODEL, NOSE_TIP_MODEL]
                    at = AffineTransformer(lms_s[BASE_LANDMARKS, :], BASE_FACE_MODEL,
                                         mp_hor_pts, mp_ver_pts, model_hor_pts, model_ver_pts)

                    # Process eyes
                    indices_left = LEFT_IRIS + ADJACENT_LEFT_EYELID_PART
                    left_eye_points = lms_s[indices_left, :]
                    left_eye_points_model = [at.to_m2(p) for p in left_eye_points]
                    self.left_detector.update(left_eye_points_model, timestamp_ms)

                    indices_right = RIGHT_IRIS + ADJACENT_RIGHT_EYELID_PART
                    right_eye_points = lms_s[indices_right, :]
                    right_eye_points_model = [at.to_m2(p) for p in right_eye_points]
                    self.right_detector.update(right_eye_points_model, timestamp_ms)

                    left_gaze_vector = None
                    right_gaze_vector = None
                    left_status = "Calibrating..."
                    right_status = "Calibrating..."

                    # Process left eye
                    if self.left_detector.center_detected:
                        left_center = at.to_m1(self.left_detector.eye_center)
                        left_pupil = lms_s[LEFT_PUPIL]
                        left_gaze_vector = left_pupil - left_center
                        left_out_of_bounds = self.is_gaze_out_of_bounds(left_gaze_vector, True)
                        left_status = "Out of bounds" if left_out_of_bounds else "Normal"
                        
                        if self.vis_options:
                            left_pupil_px = relative(left_pupil[:2], frame.shape)
                            self.draw_gaze_vector(frame, left_pupil_px, left_gaze_vector, left_out_of_bounds)

                        if left_out_of_bounds:
                            self.log_cheating_event("Gaze away - left eye", frame)

                    # Process right eye
                    if self.right_detector.center_detected:
                        right_center = at.to_m1(self.right_detector.eye_center)
                        right_pupil = lms_s[RIGHT_PUPIL]
                        right_gaze_vector = right_pupil - right_center
                        right_out_of_bounds = self.is_gaze_out_of_bounds(right_gaze_vector, False)
                        right_status = "Out of bounds" if right_out_of_bounds else "Normal"
                        
                        if self.vis_options:
                            right_pupil_px = relative(right_pupil[:2], frame.shape)
                            self.draw_gaze_vector(frame, right_pupil_px, right_gaze_vector, right_out_of_bounds)

                        if right_out_of_bounds:
                            self.log_cheating_event("Gaze away - right eye", frame)

                    # Draw status and handle callback
                    if self.vis_options:
                        self.draw_gaze_status(frame, left_status, right_status)

                    if self.callback and (left_gaze_vector is not None or right_gaze_vector is not None):
                        await self.callback(left_gaze_vector, right_gaze_vector)

                cv2.imshow('LaserGaze', frame)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()
