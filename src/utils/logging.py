import os
import datetime
import csv
import cv2
import numpy as np

# Import config
from config import settings

# --- Event Logging (Snapshots) ---
def log_cheating_event(reason, frame, timestamp):
    """Saves a snapshot frame and logs the event time and reason."""
    folder = settings.LOG_FOLDER_EVENTS
    try:
        os.makedirs(folder, exist_ok=True)
        time_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3] # Milliseconds
        # Sanitize reason for filename
        safe_reason = reason.replace(' ', '_').replace(':', '').replace('.', 'p').replace('>', 'gt').replace('<', 'lt')
        filename = os.path.join(folder, f"{time_str}_{safe_reason}.jpg")
        cv2.imwrite(filename, frame)

        # Log event to a text file
        log_filepath = os.path.join(folder, "event_log.txt")
        with open(log_filepath, "a") as log_file:
            log_file.write(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}: {reason}\n")
        # print(f"Logged event: {reason} at {timestamp}") # Optional: console confirmation
    except Exception as e:
        print(f"Error writing cheating log file/image: {e}")


# --- CSV Data Logging ---
csv_writer = None
csv_file = None

def setup_csv_logging():
    """Initializes the CSV file and writer for detailed data logging."""
    global csv_writer, csv_file
    folder = settings.LOG_FOLDER_DATA
    try:
        os.makedirs(folder, exist_ok=True)
        csv_filename = os.path.join(folder, f"gaze_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        print(f"Logging data to: {csv_filename}")
        # Use 'utf-8' encoding for broader compatibility
        csv_file = open(csv_filename, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)

        # Define header row
        header = [
            "Timestamp", "Frame", "Yaw", "Pitch", "PhoneDetectedFlag", "PhoneDetectedRaw",
            "IsAnomaly", "AnomalyScore", "CurrentOffScreenDuration", "CompletedOffScreenDuration",
            # Add features used by anomaly detector
            "Feat_Yaw", "Feat_Pitch", "Feat_YawPitchProd", "Feat_YawPitchDiff", "Feat_Phone",
            # Add face bounding box (from L2CS)
            "FaceBox_x1", "FaceBox_y1", "FaceBox_x2", "FaceBox_y2"
            ]
        # Add landmark features and raw coordinates
        for i in range(settings.NUM_LANDMARKS):
             header.extend([f"Feat_LM_{i}_x", f"Feat_LM_{i}_y"])
        for i in range(settings.NUM_LANDMARKS):
             header.extend([f"Raw_LM_{i}_x", f"Raw_LM_{i}_y"])

        csv_writer.writerow(header)
        return True # Indicate success

    except IOError as e:
        print(f"Error: Could not open CSV log file for writing: {e}")
        csv_writer = None
        csv_file = None
        return False # Indicate failure

def format_value(val, precision=4):
    """Helper to format numbers for CSV, handling None/NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "" # Represent missing values as empty strings in CSV
    if isinstance(val, (float, np.floating)):
        return f"{val:.{precision}f}"
    if isinstance(val, (int, np.integer)):
        return str(val) # Integers don't need decimal points
    return str(val) # Return string representation for other types (e.g., bool as True/False)


def write_csv_log_row(
    timestamp, frame_count, yaw_deg, pitch_deg,
    phone_detected_flag, phone_detected_raw, # Raw=this frame, Flag=consecutive
    is_anomaly, anomaly_score, anomaly_features, # Pass the features list
    current_off_screen_duration, completed_off_screen_duration,
    face_bbox, # L2CS face bbox [x1,y1,x2,y2] or None
    landmarks_xy # List of [x,y] pairs or None
    ):
    """Writes a single row of data to the CSV log file."""
    if csv_writer is None:
        return # Do nothing if logging is not set up

    try:
        log_row = [
            timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            frame_count,
            format_value(yaw_deg),
            format_value(pitch_deg),
            int(phone_detected_flag), # Log flag as 0 or 1
            int(phone_detected_raw),  # Log raw detection as 0 or 1
            int(is_anomaly),          # Log anomaly flag as 0 or 1
            format_value(anomaly_score),
            format_value(current_off_screen_duration),
            format_value(completed_off_screen_duration)
        ]

        # Add anomaly features (handle None case)
        if anomaly_features and len(anomaly_features) == (5 + settings.NUM_LANDMARKS * 2):
            log_row.extend(map(format_value, anomaly_features))
        else:
            log_row.extend([""] * (5 + settings.NUM_LANDMARKS * 2)) # Add placeholders if features missing


        # Add face bounding box (handle None)
        if face_bbox and len(face_bbox) == 4:
            log_row.extend(map(format_value, face_bbox))
        else:
            log_row.extend([""] * 4) # Add placeholders

        # Add raw landmarks (handle None)
        if landmarks_xy and len(landmarks_xy) == settings.NUM_LANDMARKS:
            flat_landmarks = [coord for pair in landmarks_xy for coord in pair]
            log_row.extend(map(format_value, flat_landmarks))
        else:
            log_row.extend([""] * (settings.NUM_LANDMARKS * 2)) # Add placeholders


        csv_writer.writerow(log_row)

        # Optional: Flush periodically to ensure data is written
        if frame_count % 100 == 0:
            csv_file.flush()

    except Exception as e:
        print(f"Error writing row to CSV: {e}")


def close_csv_logging():
    """Closes the CSV file."""
    global csv_file, csv_writer
    if csv_file:
        try:
            csv_file.close()
            print("CSV log file closed.")
        except Exception as e:
            print(f"Error closing CSV file: {e}")
    csv_file = None
    csv_writer = None