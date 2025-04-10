from pathlib import Path
import datetime
import os
import csv
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from l2cs import Pipeline, render
from l2cs.results import GazeResultContainer
from sklearn.ensemble import IsolationForest
from collections import deque

# Constants
PHONE_CLASS_ID = 67
PHONE_DETECTION_THRESHOLD = 5
FEATURE_WINDOW_SIZE = 50  # Number of samples before training/prediction
CONTAMINATION = 0.1  # Expected proportion of outliers
RETRAIN_PROBABILITY = 0.1  # Probability to retrain after detecting anomaly

class GazeAnomalyDetector:
    def __init__(self, window_size=FEATURE_WINDOW_SIZE):
        self.model = IsolationForest(contamination=CONTAMINATION, random_state=42)
        self.window_size = window_size
        self.feature_buffer = deque(maxlen=window_size)
        self.is_trained = False
        
    def add_sample(self, yaw, pitch, phone_detected):
        # Create feature vector from gaze data
        features = [yaw, pitch, abs(yaw*pitch), abs(yaw - pitch), phone_detected]
        self.feature_buffer.append(features)
        
    def is_ready(self):
        return len(self.feature_buffer) >= self.window_size
        
    def train(self):
        if not self.is_ready():
            return False
            
        X = np.array(list(self.feature_buffer))
        self.model.fit(X)
        self.is_trained = True
        return True
        
    def predict(self, yaw, pitch, phone_detected):
        if not self.is_trained:
            return False
            
        features = np.array([[yaw, pitch, abs(yaw*pitch), phone_detected]])
        prediction = self.model.predict(features)
        # -1 potential cheating
        return prediction[0] == -1

def log_cheating_event(reason, frame):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = "cheating_logs"
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"{timestamp}_{reason.replace(' ', '_')}.jpg")
    cv2.imwrite(filename, frame)
    with open(os.path.join(folder, "log.txt"), "a") as log_file:
        log_file.write(f"{timestamp}: {reason}\n")

def main():
    # Initialize models
    phone_detector = YOLO('yolo11l.pt')
    cwd = Path.cwd()
    gaze_pipeline = Pipeline(
        weights=cwd / 'models' / 'L2CSNet' / 'Gaze360' / 'L2CSNet_gaze360.pkl',
        arch='ResNet50',
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Initialize anomaly detector
    anomaly_detector = GazeAnomalyDetector()
    initial_training_done = False
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return
    
    # Initialize variables
    phone_detection_frames = 0
    
    # Setup CSV logging
    csv_filename = "gaze_data.csv"
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "yaw", "pitch", "phoneDetected", "anomaly"])
    
    print("Starting monitoring. Press 'q' to quit.")
    
    while True:
        # Capture frame
        success, frame = cap.read()
        if not success or frame is None:
            print("Failed to read from camera.")
            break
        
        frame = cv2.flip(frame, 1)
        
        # Detect phone
        detections = phone_detector.predict(frame, classes=[PHONE_CLASS_ID], stream=True, verbose=False, conf=0.3)
        phone_detected = any(det.boxes for det in detections)
        
        if phone_detected:
            phone_detection_frames += 1
            for det in detections:
                if det.boxes:
                    frame = det.plot()
        else:
            phone_detection_frames = 0
            
        # Log phone detection event if phone visible for threshold frames
        if phone_detection_frames >= PHONE_DETECTION_THRESHOLD:
            log_cheating_event("Phone detected", frame)
            phone_detection_frames = 0
        
        try:
            # Analyze gaze
            results: GazeResultContainer = gaze_pipeline.step(frame)
            
            if results.yaw:
                yaw_deg = float(results.yaw[0])
                pitch_deg = float(results.pitch[0])
                
                # Add sample to anomaly detector
                anomaly_detector.add_sample(yaw_deg, pitch_deg, int(phone_detected), )
                
                # Train the model with initial data
                if not initial_training_done and anomaly_detector.is_ready():
                    anomaly_detector.train()
                    initial_training_done = True
                    print("Initial anomaly detection model trained")
                
                # Detect anomalies
                anomaly = False
                if initial_training_done:
                    anomaly = anomaly_detector.predict(yaw_deg, pitch_deg, int(phone_detected))
                
                if anomaly:
                    log_cheating_event("Gaze anomaly detected with Isolation Forest", frame)
                    # Retrain the model periodically to adapt to student's behavior
                    if np.random.random() < RETRAIN_PROBABILITY:
                        anomaly_detector.train()
                        print("Anomaly detector retrained")
                
                # Log data to CSV
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                csv_writer.writerow([timestamp, yaw_deg, pitch_deg, int(phone_detected), int(anomaly)])
                csv_file.flush()
                
                # Display information on frame
                cv2.putText(frame, f"YAW: {yaw_deg:.1f}, PITCH: {pitch_deg:.1f}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Display anomaly status
                if anomaly:
                    cv2.putText(frame, "ANOMALY DETECTED", (20, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Render gaze visualization
                frame = render(frame, results)
            else:
                cv2.putText(frame, "No face detected", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        except ValueError:
            cv2.putText(frame, "No face detected", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display status info
        training_status = "Trained" if initial_training_done else f"Training ({len(anomaly_detector.feature_buffer)}/{FEATURE_WINDOW_SIZE})"
        cv2.putText(frame, f"Model: {training_status}", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow("Cheating Detection", frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print("Monitoring stopped.")

if __name__ == "__main__":
    main()