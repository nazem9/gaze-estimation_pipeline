import time
import datetime
import os
import csv
from dataclasses import dataclass
from pathlib import Path
from collections import deque
import random

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ultralytics import YOLO
from l2cs import Pipeline, render

from sklearn.ensemble import IsolationForest


PHONE_CLASS_ID = 67
PHONE_DETECTION_THRESHOLD = 5

CONTAMINATION = 0.05

ANGLE_HISTORY_LENGTH = 30
OFF_SCREEN_DURATION_THRESHOLD = 2.5
VISUALIZATION_UPDATE_INTERVAL = 5
PLOT_HISTORY_LENGTH = 200
OFF_SCREEN_BAR_COUNT = 15
NUM_LANDMARKS = 5
MIN_CALIBRATION_SAMPLES = 100


class GazeAnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=CONTAMINATION, random_state=42, n_estimators=150, max_samples='auto')
        self.feature_buffer = deque(maxlen=PLOT_HISTORY_LENGTH)
        self.is_trained = False
        self.anomaly_scores = deque(maxlen=PLOT_HISTORY_LENGTH)

    def _calculate_features(self, yaw, pitch, phone_detected):
        valid_gaze = yaw is not None and pitch is not None
        _yaw = yaw if valid_gaze else 0.0
        _pitch = pitch if valid_gaze else 0.0
        feature_prod = _yaw * _pitch
        feature_diff = _yaw - _pitch
        features = [_yaw, _pitch, feature_prod, feature_diff, float(phone_detected)]
        return features

    def add_sample_and_calculate_features(self, yaw, pitch, phone_detected, landmarks):
        features = self._calculate_features(yaw, pitch, phone_detected) + landmarks
        self.feature_buffer.append(features)
        return features

    def train_on_calibration(self, calibration_features):
        if not calibration_features or len(calibration_features) < MIN_CALIBRATION_SAMPLES:
            print(f"Error: Not enough calibration data ({len(calibration_features)} samples) to train anomaly detector. Need at least {MIN_CALIBRATION_SAMPLES}.")
            self.is_trained = False
            return False

        X = np.array(calibration_features)

        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("Warning: NaN or Inf found in calibration data. Skipping training.")
            self.is_trained = False
            return False

        print(f"Training Isolation Forest with {len(X)} calibration samples...")
        try:
            self.model.fit(X)
            self.is_trained = True
            print("Isolation Forest trained successfully on calibration data.")
            return True
        except ValueError as ve:
            print(f"ValueError during Isolation Forest training: {ve}")
            print("This might happen if all calibration features are identical.")
            self.is_trained = False
            return False
        except Exception as e:
            print(f"Error during Isolation Forest training: {e}")
            self.is_trained = False
            return False

    def predict(self, yaw, pitch, phone_detected):
        if not self.is_trained:
            self.anomaly_scores.append(0.0)
            return False, 0.0

        features_arr = np.array([self._calculate_features(yaw, pitch, phone_detected)])

        if np.any(np.isnan(features_arr)) or np.any(np.isinf(features_arr)):
            print("Warning: NaN or Inf found in features for prediction. Returning non-anomalous.")
            self.anomaly_scores.append(0.0)
            return False, 0.0

        try:
            prediction = self.model.predict(features_arr)
            anomaly_score = self.model.decision_function(features_arr)[0]
            self.anomaly_scores.append(anomaly_score)
            is_anomaly = (prediction[0] == -1)
            return is_anomaly, anomaly_score
        except Exception as e:
            print(f"Error during Isolation Forest prediction: {e}")
            self.anomaly_scores.append(0.0)
            return False, 0.0


class GazeDurationTracker:
    def __init__(self, yaw_threshold=25, pitch_threshold=25):
        self.yaw_threshold = abs(yaw_threshold)
        self.pitch_threshold = abs(pitch_threshold)
        self.off_screen_start_time = None
        self.off_screen_durations = deque(maxlen=OFF_SCREEN_BAR_COUNT)

    def update(self, yaw, pitch):
        completed_duration = 0
        if yaw is None or pitch is None:
            if self.off_screen_start_time is not None:
                 duration = (datetime.datetime.now() - self.off_screen_start_time).total_seconds()
                 # completed_duration = duration # Don't record duration if face lost
                 self.off_screen_start_time = None
            return completed_duration

        is_off_screen = abs(yaw) > self.yaw_threshold or abs(pitch) > self.pitch_threshold
        current_time = datetime.datetime.now()
        duration_event_completed = 0

        if is_off_screen:
            
            if self.off_screen_start_time is None:
                self.off_screen_start_time = current_time
        elif not is_off_screen and self.off_screen_start_time is not None:
            duration = (current_time - self.off_screen_start_time).total_seconds()
            self.off_screen_durations.append(duration)
            duration_event_completed = duration
            self.off_screen_start_time = None

        return duration_event_completed

    def get_current_off_screen_duration(self):
        if self.off_screen_start_time:
            return (datetime.datetime.now() - self.off_screen_start_time).total_seconds()
        return 0

    def set_thresholds(self, yaw_thresh, pitch_thresh):
        print(f"Updating off-screen thresholds: Yaw={abs(yaw_thresh):.2f}, Pitch={abs(pitch_thresh):.2f}")
        self.yaw_threshold = abs(yaw_thresh)
        self.pitch_threshold = abs(pitch_thresh)


def log_cheating_event(reason, frame, timestamp):
    folder = "cheating_logs"
    os.makedirs(folder, exist_ok=True)
    time_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = os.path.join(folder, f"{time_str}_{reason.replace(' ', '_').replace(':', '').replace('.', 'p')}.jpg")
    try:
        cv2.imwrite(filename, frame)
        log_filepath = os.path.join(folder, "event_log.txt")
        with open(log_filepath, "a") as log_file:
            log_file.write(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}: {reason}\n")
    except Exception as e:
        print(f"Error writing log file/image: {e}")


def setup_visualization():
    plt.ion()
    fig = plt.figure(figsize=(16, 12)) # Increased height for landmark plots
    gs = GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3) # Changed to 4 rows

    ax1 = fig.add_subplot(gs[0, 0])
    line_yaw, = ax1.plot([], [], label="Yaw", color="blue", lw=1)
    line_pitch, = ax1.plot([], [], label="Pitch", color="green", lw=1)
    ax1.set_title("Gaze Angles (Live)")
    ax1.set_ylabel("Degrees")
    ax1.set_xlabel("Time (s)")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=':')

    ax2 = fig.add_subplot(gs[1, 0])
    line_anomaly_score, = ax2.plot([], [], label="Anomaly Score", color="red", lw=1)
    ax2.axhline(0, color='grey', linestyle='--', lw=1, label='Approx. Threshold')
    ax2.set_title("Anomaly Score (Lower is more anomalous)")
    ax2.set_ylabel("Score")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle=':')

    ax3 = fig.add_subplot(gs[0, 1])
    line_phone, = ax3.plot([], [], label="Phone Detected", color="orange", lw=1.5, marker='.', linestyle='None')
    ax3.set_title("Phone Detection Status")
    ax3.set_ylabel("Detected (1) / Not Detected (0)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylim(-0.1, 1.1)
    ax3.legend(loc="upper left")
    ax3.grid(True, linestyle=':')

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title(f"Recent Off-Screen Events (>{OFF_SCREEN_DURATION_THRESHOLD:.1f}s)")
    ax4.set_ylabel("Duration (s)")
    ax4.set_xlabel("Event Index")
    ax4.grid(True, axis='y', linestyle=':')

    # New Landmark Plots
    landmark_colors = plt.cm.viridis(np.linspace(0, 1, NUM_LANDMARKS))
    landmark_lines = {}

    ax6 = fig.add_subplot(gs[2, 0]) # New Axes for Landmark X
    ax6.set_title("Landmark X Coordinates")
    ax6.set_ylabel("X Coordinate (pixels)")
    ax6.set_xlabel("Time (s)")
    ax6.grid(True, linestyle=':')
    for i in range(NUM_LANDMARKS):
        line, = ax6.plot([], [], label=f"LM {i} X", color=landmark_colors[i], lw=1)
        landmark_lines[f'lm{i}_x'] = line
    ax6.legend(loc="upper left", fontsize='small')


    ax7 = fig.add_subplot(gs[2, 1]) # New Axes for Landmark Y
    ax7.set_title("Landmark Y Coordinates")
    ax7.set_ylabel("Y Coordinate (pixels)")
    ax7.set_xlabel("Time (s)")
    ax7.grid(True, linestyle=':')
    for i in range(NUM_LANDMARKS):
        line, = ax7.plot([], [], label=f"LM {i} Y", color=landmark_colors[i], lw=1)
        landmark_lines[f'lm{i}_y'] = line
    ax7.legend(loc="upper left", fontsize='small')


    ax5 = fig.add_subplot(gs[3, :]) # Moved Gaze Scatter Plot to the bottom row
    scatter_gaze = ax5.scatter([], [], alpha=0.5, s=8, label="Gaze Points (Recent)")
    ax5.set_title("Gaze Distribution (Yaw vs Pitch) & Thresholds")
    ax5.set_xlabel("Yaw (degrees)")
    ax5.set_ylabel("Pitch (degrees)")
    ax5.grid(True, linestyle=':')
    ax5.axhline(0, color='grey', lw=0.5)
    ax5.axvline(0, color='grey', lw=0.5)
    line_yaw_thresh_pos, = ax5.plot([], [], 'r--', lw=1, label='Yaw Thresh')
    line_yaw_thresh_neg, = ax5.plot([], [], 'r--', lw=1)
    line_pitch_thresh_pos, = ax5.plot([], [], 'g--', lw=1, label='Pitch Thresh')
    line_pitch_thresh_neg, = ax5.plot([], [], 'g--', lw=1)
    ax5.legend(loc='upper right')

    plt.tight_layout(pad=2.0)

    visualization_objects = {
        'fig': fig,
        'axes': [ax1, ax2, ax3, ax4, ax5, ax6, ax7], # Added ax6, ax7
        'lines': {
            'yaw': line_yaw,
            'pitch': line_pitch,
            'anomaly_score': line_anomaly_score,
            'phone': line_phone,
            'yaw_thresh_pos': line_yaw_thresh_pos,
            'yaw_thresh_neg': line_yaw_thresh_neg,
            'pitch_thresh_pos': line_pitch_thresh_pos,
            'pitch_thresh_neg': line_pitch_thresh_neg,
            **landmark_lines # Added landmark lines
        },
        'scatter': scatter_gaze,
        'bar_container': None
    }
    return visualization_objects


def update_visualization(vis_objects, data_queues, frame_count, tracker, anomaly_detector):
    if frame_count % VISUALIZATION_UPDATE_INTERVAL != 0:
        return

    times = list(data_queues['time'])
    if not times:
        return

    vis_objects['lines']['yaw'].set_data(times, list(data_queues['yaw']))
    vis_objects['lines']['pitch'].set_data(times, list(data_queues['pitch']))
    vis_objects['lines']['anomaly_score'].set_data(times[-len(anomaly_detector.anomaly_scores):], list(anomaly_detector.anomaly_scores))
    vis_objects['lines']['phone'].set_data(times, list(data_queues['phone_detected']))

    # Update landmark lines
    for i in range(NUM_LANDMARKS):
        if f'lm{i}_x' in vis_objects['lines']:
            vis_objects['lines'][f'lm{i}_x'].set_data(times, list(data_queues[f'lm{i}_x']))
        if f'lm{i}_y' in vis_objects['lines']:
            vis_objects['lines'][f'lm{i}_y'].set_data(times, list(data_queues[f'lm{i}_y']))


    valid_gaze_indices = [i for i, (y, p) in enumerate(zip(data_queues['yaw'], data_queues['pitch'])) if y is not None and p is not None]
    valid_yaw = [data_queues['yaw'][i] for i in valid_gaze_indices]
    valid_pitch = [data_queues['pitch'][i] for i in valid_gaze_indices]
    if valid_yaw:
        vis_objects['scatter'].set_offsets(np.column_stack((valid_yaw, valid_pitch)))
        ax5 = vis_objects['axes'][4] # Gaze scatter is now ax5 (index 4)
        max_data_x = max(np.max(np.abs(valid_yaw)), tracker.yaw_threshold) if valid_yaw else tracker.yaw_threshold
        max_data_y = max(np.max(np.abs(valid_pitch)), tracker.pitch_threshold) if valid_pitch else tracker.pitch_threshold
        ax5.set_xlim(-max_data_x * 1.2, max_data_x * 1.2)
        ax5.set_ylim(-max_data_y * 1.2, max_data_y * 1.2)

    ax4 = vis_objects['axes'][3]
    recent_durations = list(tracker.off_screen_durations)
    valid_durations = [d for d in recent_durations if d >= OFF_SCREEN_DURATION_THRESHOLD]
    positions = np.arange(len(valid_durations))

    ax4.cla()
    ax4.set_title(f"Recent Off-Screen Events (>{OFF_SCREEN_DURATION_THRESHOLD:.1f}s)")
    ax4.set_ylabel("Duration (s)")
    ax4.set_xlabel("Event Index")
    ax4.grid(True, axis='y', linestyle=':')
    if len(valid_durations) > 0:
        vis_objects['bar_container'] = ax4.bar(positions, valid_durations, color="teal")
        ax4.set_xticks(positions)
        ax4.set_xlim(left=-0.5, right=max(len(valid_durations) - 0.5, OFF_SCREEN_BAR_COUNT - 0.5))
        max_duration = max(valid_durations) if valid_durations else 5
        ax4.set_ylim(bottom=0, top=max_duration * 1.1)
    else:
        ax4.set_xticks([])
        ax4.set_ylim(bottom=0, top=5)

    yaw_thresh = tracker.yaw_threshold
    pitch_thresh = tracker.pitch_threshold
    ax5 = vis_objects['axes'][4] # Gaze scatter is now ax5 (index 4)
    xlim = ax5.get_xlim()
    ylim = ax5.get_ylim()
    vis_objects['lines']['yaw_thresh_pos'].set_data([yaw_thresh, yaw_thresh], ylim)
    vis_objects['lines']['yaw_thresh_neg'].set_data([-yaw_thresh, -yaw_thresh], ylim)
    vis_objects['lines']['pitch_thresh_pos'].set_data(xlim, [pitch_thresh, pitch_thresh])
    vis_objects['lines']['pitch_thresh_neg'].set_data(xlim, [-pitch_thresh, -pitch_thresh])
    vis_objects['lines']['yaw_thresh_pos'].set_ydata(ylim)
    vis_objects['lines']['yaw_thresh_neg'].set_ydata(ylim)
    vis_objects['lines']['pitch_thresh_pos'].set_xdata(xlim)
    vis_objects['lines']['pitch_thresh_neg'].set_xdata(xlim)

    min_time = times[0] if times else 0
    max_time = times[-1] if times else 1
    # Update axes limits for all time-based plots (indices 0, 1, 2, 5, 6)
    for i in [0, 1, 2, 5, 6]:
        vis_objects['axes'][i].relim()
        vis_objects['axes'][i].autoscale_view(scalex=False, scaley=True)
        vis_objects['axes'][i].set_xlim(min_time, max_time)

    try:
        vis_objects['fig'].canvas.draw_idle()
        vis_objects['fig'].canvas.flush_events()
    except Exception as e:
        print(f"Plotting Error: {e}")


def enhanced_calibration_mode(gaze_pipeline, cap, anomaly_detector, duration_per_target=5):
    print("\n--- Starting Enhanced Calibration (Fullscreen) ---")
    print(f"Please follow the instructions. Look at the markers when they appear.")
    print("Press 'q' at any time to abort.")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Calibration using frame dimensions: {frame_w}x{frame_h}")

    calibration_features = []
    calibration_raw_yaw = []
    calibration_raw_pitch = []

    targets = {
        "Center": (0.5, 0.5), "Top-Left": (0.1, 0.1), "Top-Right": (0.9, 0.1),
        "Bottom-Left": (0.1, 0.9), "Bottom-Right": (0.9, 0.9), "Mid-Left": (0.05, 0.5),
        "Mid-Right": (0.95, 0.5), "Mid-Top": (0.5, 0.05), "Mid-Bottom": (0.5, 0.95),
    }
    target_radius = 25
    target_color = (0, 255, 255)
    text_color = (255, 255, 255)
    info_color = (50, 200, 250)

    window_name = "Calibration"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print("Calibration window set to fullscreen. Press 'q' to exit.")

    for target_name, (tx_rel, ty_rel) in targets.items():
        tx_px = int(tx_rel * frame_w)
        ty_px = int(ty_rel * frame_h)

        print(f"Calibration step: Look at the '{target_name}' marker.")
        start_time = time.time()
        samples_this_target = 0

        while time.time() - start_time < duration_per_target:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            cv2.circle(display_frame, (tx_px, ty_px), target_radius, target_color, -1)
            cv2.putText(display_frame, f"Look at: {target_name}", (frame_w // 2 - 150, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
            remaining = duration_per_target - (time.time() - start_time)
            cv2.putText(display_frame, f"Time left: {remaining:.1f}s", (frame_w // 2 - 100, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

            current_yaw, current_pitch = None, None
            try:
                results = gaze_pipeline.step(display_frame)
                if results and results.pitch is not None and len(results.pitch) > 0:
                    current_yaw = float(results.yaw[0])
                    current_pitch = float(results.pitch[0])

                    calibration_raw_yaw.append(current_yaw)
                    calibration_raw_pitch.append(current_pitch)

                    features = anomaly_detector._calculate_features(current_yaw, current_pitch, False)
                    calibration_features.append(features)
                    samples_this_target += 1

                    cv2.putText(display_frame, f"Yaw: {current_yaw:.1f}", (20, frame_h - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_color, 2)
                    cv2.putText(display_frame, f"Pitch: {current_pitch:.1f}", (20, frame_h - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_color, 2)
                else:
                    cv2.putText(display_frame, "No Face Detected", (20, frame_h - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            except ValueError:
                 cv2.putText(display_frame, "No Face Detected", (20, frame_h - 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            except Exception as e:
                 print(f"Error during calibration gaze step: {e}")
                 cv2.putText(display_frame, "Gaze Error", (20, frame_h - 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Calibration aborted by user.")
                cv2.destroyWindow(window_name)
                return None, None

        print(f"  Collected {samples_this_target} samples for {target_name}.")

    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, "Calibration Complete!", (frame_w // 2 - 200, frame_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow(window_name, frame)
        cv2.waitKey(2000)

    cv2.destroyWindow(window_name)
    print("--- Calibration Data Collection Complete ---")

    if len(calibration_features) < MIN_CALIBRATION_SAMPLES:
        print(f"Warning: Only collected {len(calibration_features)} valid samples (Min required: {MIN_CALIBRATION_SAMPLES}).")
        print("Anomaly detection training may fail or be unreliable.")
        thresholds = {'yaw': 25.0, 'pitch': 25.0}
    else:
         print(f"Collected a total of {len(calibration_features)} valid samples for anomaly training.")
         if not calibration_raw_yaw or not calibration_raw_pitch:
             print("Warning: No raw gaze data collected despite samples > min. Using default thresholds.")
             thresholds = {'yaw': 25.0, 'pitch': 25.0}
         else:
             try:
                 yaw_q_low, yaw_q_high = np.percentile(calibration_raw_yaw, [5, 95])
                 pitch_q_low, pitch_q_high = np.percentile(calibration_raw_pitch, [5, 95])
                 filtered_yaw = [y for y in calibration_raw_yaw if yaw_q_low <= y <= yaw_q_high]
                 filtered_pitch = [p for p in calibration_raw_pitch if pitch_q_low <= p <= pitch_q_high]

                 if not filtered_yaw or not filtered_pitch:
                      print("Warning: Could not determine gaze range after filtering outliers. Using defaults.")
                      thresholds = {'yaw': 25.0, 'pitch': 25.0}
                 else:
                     max_abs_yaw = np.max(np.abs(filtered_yaw))
                     max_abs_pitch = np.max(np.abs(filtered_pitch))
                     yaw_threshold = max_abs_yaw + 5
                     pitch_threshold = max_abs_pitch + 8

                     print("Calibration-based Off-Screen Thresholds:")
                     print(f"  Filtered Yaw Range (5-95%): [{np.min(filtered_yaw):.1f}, {np.max(filtered_yaw):.1f}]")
                     print(f"  Filtered Pitch Range (5-95%): [{np.min(filtered_pitch):.1f}, {np.max(filtered_pitch):.1f}]")
                     print(f"  Calculated Thresholds -> Yaw: {yaw_threshold:.2f}, Pitch: {pitch_threshold:.2f}")
                     thresholds = {'yaw': yaw_threshold, 'pitch': pitch_threshold}
             except Exception as e:
                  print(f"Error calculating thresholds: {e}. Using defaults.")
                  thresholds = {'yaw': 25.0, 'pitch': 25.0}

    return calibration_features, thresholds


def main():
    print("Loading models...")
    try:
        phone_detector = YOLO('yolo11l.pt')
        print("YOLO model ('yolo11l.pt') loaded.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}. Ensure 'yolo11l.pt' is available.")
        return

    l2cs_model_path = Path.cwd() / "models"/ "L2CSNet" / "Gaze360" / "L2CSNet_gaze360.pkl"
    if not l2cs_model_path.exists():
         l2cs_model_path_alt = Path(__file__).parent / "models" / "L2CSNet_gaze360.pkl"
         if l2cs_model_path_alt.exists():
             l2cs_model_path = l2cs_model_path_alt
         else:
            print(f"Error: L2CS model weights not found at:")
            print(f"  - {l2cs_model_path}")
            print(f"  - {l2cs_model_path_alt}")
            print("Please ensure the L2CS model file exists in the 'models' subdirectory relative to your script or CWD.")
            return

    try:
        gaze_pipeline = Pipeline(
            weights=l2cs_model_path,
            arch='ResNet50',
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        print(f"L2CS Gaze Pipeline loaded on device: {gaze_pipeline.device}")
    except Exception as e:
        print(f"Error loading L2CS Gaze Pipeline: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Webcam opened.")

    anomaly_detector = GazeAnomalyDetector()

    calibration_features, off_screen_thresholds = enhanced_calibration_mode(
        gaze_pipeline, cap, anomaly_detector, duration_per_target=5
    )

    anomaly_training_successful = False
    if calibration_features:
        anomaly_training_successful = anomaly_detector.train_on_calibration(calibration_features)

    if not anomaly_training_successful:
        print("*****************************************************")
        print("Anomaly detector training failed or insufficient data.")
        print("Anomaly detection will be DISABLED.")
        print("*****************************************************")

    if off_screen_thresholds:
        gaze_duration_tracker = GazeDurationTracker(
            yaw_threshold=off_screen_thresholds['yaw'],
            pitch_threshold=off_screen_thresholds['pitch']
        )
    else:
        print("Calibration failed to provide thresholds, using defaults for off-screen tracker.")
        gaze_duration_tracker = GazeDurationTracker(yaw_threshold=25, pitch_threshold=25)

    phone_detected_flag = False
    phone_consecutive_frames = 0
    frame_count = 0

    log_folder = "logs"
    os.makedirs(log_folder, exist_ok=True)
    csv_filename = os.path.join(log_folder, f"gaze_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    print(f"Logging data to: {csv_filename}")
    try:
        csv_file = open(csv_filename, 'w', newline='')
        csv_writer = csv.writer(csv_file)
    except IOError as e:
        print(f"Error: Could not open CSV log file for writing: {e}")
        print("CSV logging disabled.")
        csv_writer = None

    if csv_writer:
        header = ["Timestamp", "Frame", "Yaw", "Pitch", "PhoneDetected",
                  "IsAnomaly", "AnomalyScore", "CurrentOffScreenDuration", "CompletedOffScreenDuration",
                  "Feature_Yaw", "Feature_Pitch", "Feature_YawPitchProd", "Feature_YawPitchDiff", "Feature_Phone",
                  "Bbox_x1", "Bbox_y1", "Bbox_x2", "Bbox_y2"]
        for i in range(NUM_LANDMARKS):
            header.extend([f"lm_{i}_x", f"lm_{i}_y"])
        csv_writer.writerow(header)

    vis_objects = setup_visualization()

    visualization_data = {
        'time': deque(maxlen=PLOT_HISTORY_LENGTH),
        'yaw': deque(maxlen=PLOT_HISTORY_LENGTH),
        'pitch': deque(maxlen=PLOT_HISTORY_LENGTH),
        'phone_detected': deque(maxlen=PLOT_HISTORY_LENGTH),
    }
    # Add deques for landmark data
    for i in range(NUM_LANDMARKS):
        visualization_data[f'lm{i}_x'] = deque(maxlen=PLOT_HISTORY_LENGTH)
        visualization_data[f'lm{i}_y'] = deque(maxlen=PLOT_HISTORY_LENGTH)


    start_time_monitoring = datetime.datetime.now()
    last_vis_time = time.time()
    accumulated_frames_fps = 0

    print("\n--- Starting Monitoring ---")
    print("Press 'q' in the display window to quit.")
    print(f"Anomaly Detection Active: {anomaly_detector.is_trained}")
    print(f"Off-Screen Thresholds: Yaw={gaze_duration_tracker.yaw_threshold:.1f}, Pitch={gaze_duration_tracker.pitch_threshold:.1f}")

    while True:
        success, frame = cap.read()
        current_timestamp = datetime.datetime.now()
        if not success or frame is None:
            print("Warning: Failed to read frame from camera.")
            time.sleep(0.1)
            continue

        frame_count += 1
        accumulated_frames_fps += 1
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        results_yolo = phone_detector.predict(frame, classes=[PHONE_CLASS_ID], verbose=False, conf=0.45, device=gaze_pipeline.device)

        phone_bboxes = []
        if results_yolo and len(results_yolo) > 0:
             res = results_yolo[0]
             if hasattr(res, 'boxes') and res.boxes is not None:
                  cpu_boxes = res.boxes.cpu()
                  for i in range(len(cpu_boxes.cls)):
                      if int(cpu_boxes.cls[i]) == PHONE_CLASS_ID:
                          box = cpu_boxes.xyxy[i].numpy()
                          phone_bboxes.append(box)
                          x1, y1, x2, y2 = map(int, box)
                          cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                          cv2.putText(display_frame, "Phone", (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        phone_detected_this_frame = len(phone_bboxes) > 0

        if phone_detected_this_frame:
            phone_consecutive_frames += 1
        else:
            phone_consecutive_frames = 0

        phone_detected_flag = phone_consecutive_frames >= PHONE_DETECTION_THRESHOLD
        if phone_consecutive_frames == PHONE_DETECTION_THRESHOLD:
            print(f"Event: Phone detected for {PHONE_DETECTION_THRESHOLD} consecutive frames.")
            log_cheating_event("Phone_Detected", frame, current_timestamp)

        yaw_deg, pitch_deg = None, None
        landmarks_flat = [np.nan] * (NUM_LANDMARKS * 2)
        landmarks_xy = [[np.nan, np.nan]] * NUM_LANDMARKS # Store as pairs
        bbox = [np.nan] * 4
        gaze_results = None

        try:
            gaze_results = gaze_pipeline.step(frame)
            if gaze_results and gaze_results.pitch is not None and len(gaze_results.pitch) > 0:
                yaw_deg = float(gaze_results.yaw[0])
                pitch_deg = float(gaze_results.pitch[0])

                if gaze_results.landmarks is not None and len(gaze_results.landmarks) > 0:
                    lm = gaze_results.landmarks[0] # Should be (5, 2)
                    if lm.shape == (NUM_LANDMARKS, 2):
                         landmarks_flat = lm.flatten().tolist()
                         landmarks_xy = lm.tolist() # Keep as pairs

                if gaze_results.bboxes is not None and len(gaze_results.bboxes) > 0:
                     bbox = gaze_results.bboxes[0].astype(int).tolist()

                display_frame = render(display_frame, gaze_results)
                if gaze_results.landmarks is not None and len(gaze_results.landmarks) > 0:
                    for (x, y) in gaze_results.landmarks[0]:
                        cv2.circle(display_frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        except ValueError:
             pass
        except Exception as e:
            print(f"Error during gaze estimation: {e}")

        current_features = anomaly_detector.add_sample_and_calculate_features(yaw_deg, pitch_deg, phone_detected_flag, landmarks = gaze_results.landmarks[0].flatten().tolist() if gaze_results.landmarks is not None else None)

        anomaly_detected = False
        anomaly_score = 0.0

        if anomaly_detector.is_trained:
            anomaly_detected, anomaly_score = anomaly_detector.predict(yaw_deg, pitch_deg, phone_detected_flag)
            if anomaly_detected:
                log_cheating_event(f"Gaze_Anomaly_Score_{anomaly_score:.2f}", frame, current_timestamp)
        else:
             anomaly_detector.anomaly_scores.append(0.0)

        completed_off_screen_duration = gaze_duration_tracker.update(yaw_deg, pitch_deg)
        current_off_screen_duration = gaze_duration_tracker.get_current_off_screen_duration()

        if completed_off_screen_duration >= OFF_SCREEN_DURATION_THRESHOLD:
            print(f"Event: Looked away for {completed_off_screen_duration:.1f} seconds.")
            log_cheating_event(f"OffScreen_{completed_off_screen_duration:.1f}s", frame, current_timestamp)

        current_time_relative = (current_timestamp - start_time_monitoring).total_seconds()
        visualization_data['time'].append(current_time_relative)
        visualization_data['yaw'].append(yaw_deg if yaw_deg is not None else np.nan)
        visualization_data['pitch'].append(pitch_deg if pitch_deg is not None else np.nan)
        visualization_data['phone_detected'].append(float(phone_detected_flag))

        # Append landmark data (or NaN if not detected)
        for i in range(NUM_LANDMARKS):
            visualization_data[f'lm{i}_x'].append(landmarks_xy[i][0])
            visualization_data[f'lm{i}_y'].append(landmarks_xy[i][1])


        if csv_writer:
            if len(bbox) != 4: bbox = [np.nan]*4
            if len(landmarks_flat) != NUM_LANDMARKS*2: landmarks_flat = [np.nan]*(NUM_LANDMARKS*2)

            log_row = [
                current_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                frame_count,
                f"{yaw_deg:.4f}" if yaw_deg is not None else np.nan,
                f"{pitch_deg:.4f}" if pitch_deg is not None else np.nan,
                int(phone_detected_flag),
                int(anomaly_detected),
                f"{anomaly_score:.4f}" if anomaly_detector.is_trained else np.nan,
                f"{current_off_screen_duration:.4f}",
                f"{completed_off_screen_duration:.4f}" if completed_off_screen_duration > 0 else 0.0,
            ]
            log_row.extend(map(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and not np.isnan(x) else np.nan, current_features))
            log_row.extend(map(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and not np.isnan(x) else np.nan, bbox))
            log_row.extend(map(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and not np.isnan(x) else np.nan, landmarks_flat))

            try:
                csv_writer.writerow(log_row)
                if frame_count % 100 == 0:
                    csv_file.flush()
            except Exception as e:
                print(f"Error writing to CSV: {e}")

        now = time.time()
        elapsed = now - last_vis_time
        fps = accumulated_frames_fps / elapsed if elapsed > 0 else 0
        if elapsed > 1.0:
            last_vis_time = now
            accumulated_frames_fps = 0

        cv2.putText(display_frame, f"FPS: {fps:.1f}", (display_frame.shape[1] - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        y_offset = 30
        if yaw_deg is not None:
            cv2.putText(display_frame, f"Yaw: {yaw_deg:.1f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2); y_offset += 25
            cv2.putText(display_frame, f"Pitch: {pitch_deg:.1f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2); y_offset += 25
        else:
            cv2.putText(display_frame, "No Face Detected (Gaze)", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2); y_offset += 25

        if anomaly_detector.is_trained:
            status_text = "ANOMALY DETECTED" if anomaly_detected else "Normal"
            status_color = (0, 0, 255) if anomaly_detected else (0, 255, 0)
            cv2.putText(display_frame, f"Anomaly: {status_text} ({anomaly_score:.2f})", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2); y_offset += 25
        else:
             cv2.putText(display_frame, "Anomaly Detection: INACTIVE", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2); y_offset += 25

        if current_off_screen_duration > 0.1:
            cv2.putText(display_frame, f"OFF-SCREEN: {current_off_screen_duration:.1f}s", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 150, 255), 2); y_offset += 25
        elif completed_off_screen_duration > 0:
             cv2.putText(display_frame, f"Returned to screen ({completed_off_screen_duration:.1f}s ago)", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100), 2); y_offset += 25

        if phone_detected_flag:
             cv2.putText(display_frame, "PHONE DETECTED", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2); y_offset += 25

        try:
            update_visualization(vis_objects, visualization_data, frame_count, gaze_duration_tracker, anomaly_detector)
        except Exception as e:
             print(f"Error updating visualization: {e}")

        cv2.imshow("Cheating Detection Monitor", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("'q' pressed, exiting.")
            break

    print("\nCleaning up resources...")
    cap.release()
    if csv_writer:
        csv_file.close()
    plt.close('all')
    cv2.destroyAllWindows()
    print("Monitoring stopped.")


if __name__ == "__main__":
    if not Path("models").is_dir():
        print("Warning: 'models' directory not found in the current working directory.")
        print("L2CS model loading might fail if the path isn't absolute or correctly relative.")
    main()