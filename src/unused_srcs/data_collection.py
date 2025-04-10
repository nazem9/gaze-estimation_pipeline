import os
import csv
import cv2
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from l2cs import Pipeline, render
from l2cs.results import GazeResultContainer

PHONE_CLASS_ID = 67
PHONE_DETECTION_THRESHOLD = 5
ANGLE_HISTORY_LENGTH = 30
Z_SCORE_THRESHOLD = 2.0

def dynamic_gaze_anomaly(history, current_yaw, current_pitch, z_threshold=Z_SCORE_THRESHOLD):
    if len(history) < 5:
        return False
    yaws, pitches = zip(*history)
    yaw_mean = np.mean(yaws)
    yaw_std = np.std(yaws)
    pitch_mean = np.mean(pitches)
    pitch_std = np.std(pitches)
    if yaw_std == 0 or pitch_std == 0:
        return False
    return abs(current_yaw - yaw_mean) > z_threshold * yaw_std or abs(current_pitch - pitch_mean) > z_threshold * pitch_std

def log_cheating_event(reason, frame):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = "cheating_logs"
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"{timestamp}_{reason.replace(' ', '_')}.jpg")
    cv2.imwrite(filename, frame)
    with open(os.path.join(folder, "log.txt"), "a") as log_file:
        log_file.write(f"{timestamp}: {reason}\n")

def main():
    phone_detector = YOLO('yolo11l.pt')
    cwd = Path.cwd()
    gaze_pipeline = Pipeline(
        weights=cwd / 'models' / 'L2CSNet' / 'Gaze360' / 'L2CSNet_gaze360.pkl',
        arch='ResNet50',
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return
    phone_detection_frames = 0
    gaze_angle_history = []
    csv_filename = "gaze_data.csv"
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "yaw", "pitch", "phoneDetected", "anomaly"])
    time_series, yaw_series, pitch_series, anomaly_series, prod_series = [], [], [], [], []
    plt.ion()
    fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(10, 8))
    line_yaw, = ax1.plot([], [], label="Yaw", color="blue")
    line_pitch, = ax1.plot([], [], label="Pitch", color="green")
    ax1.set_ylabel("Degrees")
    ax1.legend(loc="upper left")
    line_anomaly, = ax2.plot([], [], label="Anomaly", color="red")
    ax2.set_ylabel("Anomaly")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend(loc="upper left")

    ax3.set_title("Yaw x Pitch")
    line_product, = ax3.plot([], [], label="Yaw x Pitch", color="purple")
    ax3.set_xlabel("Yaw")
    start_time = datetime.datetime.now()
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            print("Failed to read from camera.")
            break
        frame = cv2.flip(frame, 1)
        detections = phone_detector.predict(frame, classes=[PHONE_CLASS_ID], stream=True, verbose=False, conf=0.3)
        phone_detected = any(det.boxes for det in detections)
        if phone_detected:
            phone_detection_frames += 1
            for det in detections:
                if det.boxes:
                    frame = det.plot()
        else:
            phone_detection_frames = 0
        if phone_detection_frames >= PHONE_DETECTION_THRESHOLD:
            log_cheating_event("Phone detected", frame)
            phone_detection_frames = 0
        try:
            results: GazeResultContainer = gaze_pipeline.step(frame)
            if results.yaw:
                yaw_deg = float(results.yaw[0])
                pitch_deg = float(results.pitch[0])
                gaze_angle_history.append((yaw_deg, pitch_deg))
                if len(gaze_angle_history) > ANGLE_HISTORY_LENGTH:
                    gaze_angle_history = gaze_angle_history[-ANGLE_HISTORY_LENGTH:]
                anomaly_flag = dynamic_gaze_anomaly(gaze_angle_history, yaw_deg, pitch_deg)
                if anomaly_flag:
                    log_cheating_event("Gaze anomaly detected", frame)
                    gaze_angle_history = []
                current_time = (datetime.datetime.now() - start_time).total_seconds()
                csv_writer.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                     yaw_deg, pitch_deg, int(phone_detected), int(anomaly_flag)])
                csv_file.flush()
                time_series.append(current_time)
                yaw_series.append(yaw_deg)
                pitch_series.append(pitch_deg)
                anomaly_series.append(1 if anomaly_flag else 0)
                prod_series.append(yaw_deg * pitch_deg)
                line_yaw.set_data(time_series, yaw_series)
                line_pitch.set_data(time_series, pitch_series)
                ax1.relim()
                ax1.autoscale_view()
                line_anomaly.set_data(time_series, anomaly_series)
                ax2.relim()
                ax2.autoscale_view()
                line_product.set_data(time_series, [y * p for y, p in zip(yaw_series, pitch_series)])
                ax3.relim()
                ax3.autoscale_view()
                ax3.set_xlim(min(yaw_series) - 10, max(yaw_series) + 10)
                plt.pause(0.001)
                cv2.putText(frame, f"YAW: {yaw_deg:.1f}, PITCH: {pitch_deg:.1f}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                frame = render(frame, results)
            else:
                cv2.putText(frame, "No face detected", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        except ValueError:
            cv2.putText(frame, "No face detected", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Cheating Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    csv_file.close()
    plt.ioff()
    plt.show()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
