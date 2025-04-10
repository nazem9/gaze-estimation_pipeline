import cv2
import torch
import time
import datetime
from pathlib import Path
import numpy as np
from collections import deque

# --- Import Configuration ---
from config import settings

# --- Import Core Components ---
from anomaly_detection.detector import GazeAnomalyDetector
from gaze_tracking.tracker import GazeDurationTracker
from gaze_tracking.calibration import enhanced_calibration_mode
from object_detection.phone_detector import PhoneDetector # Using the wrapper

# --- Import Visualization ---
from visualization.plotter import setup_visualization, update_visualization, close_plots
from visualization.frame_renderer import render_overlays

# --- Import Utilities ---
from utils.logging import setup_csv_logging, write_csv_log_row, close_csv_logging, log_cheating_event

# --- Import L2CS Pipeline ---
from l2cs import Pipeline
import argparse

def initialize_models():
    """Loads and initializes YOLO and L2CS models."""
    print("--- Initializing Models ---")
    # Phone Detector (YOLO)
    phone_detector = PhoneDetector(settings.YOLO_MODEL_PATH, settings.DEVICE)
    if phone_detector.model is None:
        # Decide if you want to continue without phone detection or exit
        print("Warning: Proceeding without phone detection.")
        # exit() # Or uncomment to stop if YOLO fails

    # Gaze Estimator (L2CS)
    l2cs_model_path = settings.L2CS_MODEL_DIR / settings.L2CS_MODEL_NAME
    gaze_pipeline = None
    if not l2cs_model_path.exists():
        # Try alternative common location (relative to script if models/ isn't top-level)
        alt_path = Path(__file__).parent.parent / "models" / "L2CSNet" / "Gaze360" / settings.L2CS_MODEL_NAME
        if alt_path.exists():
            l2cs_model_path = alt_path
        else:
             print(f"FATAL ERROR: L2CS model weights not found at expected locations:")
             print(f"  - {settings.L2CS_MODEL_DIR / settings.L2CS_MODEL_NAME}")
             print(f"  - {alt_path}")
             print("Please ensure the L2CS model file exists.")
             exit() # Cannot proceed without gaze model

    try:
        print(f"Loading L2CS model from: {l2cs_model_path}")
        gaze_pipeline = Pipeline(
            weights=l2cs_model_path,
            arch=settings.L2CS_ARCH,
            device=settings.DEVICE # Use device from config
        )
        print(f"L2CS Gaze Pipeline loaded on device: {settings.DEVICE}")
    except Exception as e:
        print(f"FATAL ERROR loading L2CS Gaze Pipeline: {e}")
        exit()

    print("--- Model Initialization Complete ---")
    return phone_detector, gaze_pipeline


def initialize_camera():
    """Opens the webcam."""
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0) # TODO: Make camera index configurable?
    if not cap.isOpened():
        print("FATAL ERROR: Could not open webcam.")
        exit()
    # Set desired resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Webcam opened successfully.")
    return cap


def main_loop(cap, phone_detector, gaze_pipeline, anomaly_detector, gaze_duration_tracker, vis_objects, vis_data):
    """The main processing loop for video frames."""
    frame_count = 0
    start_time_monitoring = datetime.datetime.now()
    last_fps_time = time.time()
    accumulated_frames_fps = 0

    print("\n--- Starting Monitoring ---")
    print(f"Interactive Plots: {'Enabled' if settings.ENABLE_INTERACTIVE_PLOTS else 'Disabled'}")
    print(f"Render Landmarks: {'Yes' if settings.RENDER_LANDMARKS_ON_FRAME else 'No'}")
    print(f"Render Gaze Vector: {'Yes' if settings.RENDER_GAZE_VECTOR_ON_FRAME else 'No'}")
    print(f"Render Info Text: {'Yes' if settings.RENDER_INFO_TEXT_ON_FRAME else 'No'}")
    print(f"Anomaly Detection Active: {anomaly_detector.is_trained}")
    print(f"Off-Screen Thresholds: Yaw={gaze_duration_tracker.yaw_threshold:.1f}, Pitch={gaze_duration_tracker.pitch_threshold:.1f}")
    print("Press 'q' in the display window to quit.")

    while True:
        success, frame = cap.read()
        current_timestamp = datetime.datetime.now()
        if not success or frame is None:
            print("Warning: Failed to read frame from camera. Skipping.")
            time.sleep(0.1) # Avoid busy-waiting if camera fails
            continue

        frame_count += 1
        accumulated_frames_fps += 1
        frame = cv2.flip(frame, 1) # Flip horizontally for intuitive view
        display_frame = frame.copy() # Create a copy for drawing overlays

        # 1. Phone Detection
        phone_detected_flag, phone_bboxes_this_frame, phone_triggered_now = phone_detector.detect(frame)
        if phone_triggered_now: # Log only when the flag becomes True
            print(f"Event: Phone detected for {settings.PHONE_DETECTION_CONSECUTIVE_FRAMES} consecutive frames.")
            log_cheating_event("Phone_Detected_ThresholdMet", frame, current_timestamp)

        # 2. Gaze Estimation
        yaw_deg, pitch_deg = None, None
        landmarks_xy = None # Store as list of [x,y] pairs
        landmarks_flat = None # Store flattened for features
        face_bbox = None # Store face bbox [x1, y1, x2, y2]
        gaze_results = None # Store the raw results object
        try:
            gaze_results = gaze_pipeline.step(frame) # Process original frame
            if gaze_results.landmarks.shape[0] > 1:
                log_cheating_event("Detected more than 1 people in the frae", frame, current_timestamp)
            if gaze_results and gaze_results.pitch is not None and len(gaze_results.pitch) > 0:
                # We have a successful gaze estimation
                yaw_deg = float(gaze_results.yaw[0])
                pitch_deg = float(gaze_results.pitch[0])
                
                # Extract landmarks if available and correctly shaped
                if gaze_results.landmarks is not None and len(gaze_results.landmarks) > 0:
                    lm_np = gaze_results.landmarks[0] # Shape should be (NUM_LANDMARKS, 2)
                    if lm_np.shape == (settings.NUM_LANDMARKS, 2):
                         landmarks_xy = lm_np.tolist() # Keep as list of pairs for rendering/logging
                         landmarks_flat = lm_np.flatten().tolist() # Flatten for features

                # Extract face bounding box if available
                if gaze_results.bboxes is not None and len(gaze_results.bboxes) > 0:
                     bbox_raw = gaze_results.bboxes[0] # Usually [x_min, y_min, x_max, y_max, score]
                     if len(bbox_raw) >= 4:
                         face_bbox = bbox_raw[:4].astype(int).tolist() # Take first 4, convert to int list

        except ValueError:
             pass # L2CS raises ValueError if no face detected, expected behavior
        except Exception as e:
            print(f"Error during gaze estimation step: {e}") # Log unexpected errors


        current_features = anomaly_detector.prepare_features_for_prediction(
            yaw_deg, pitch_deg, phone_detected_flag, landmarks_flat if settings.INCLUDE_LANDMARKS_IN_FEATURES else None
        )

        anomaly_detected, anomaly_score = anomaly_detector.predict() # Predict based on stored features
        if anomaly_detected:
            # Log anomaly event (consider adding a cooldown?)
            log_cheating_event(f"Gaze_Anomaly_Score_{anomaly_score:.2f}", frame, current_timestamp)


        # 4. Gaze Duration Tracking
        completed_off_screen_duration = gaze_duration_tracker.update(yaw_deg, pitch_deg)
        current_off_screen_duration = gaze_duration_tracker.get_current_off_screen_duration()

        if completed_off_screen_duration >= settings.OFF_SCREEN_DURATION_THRESHOLD:
            print(f"Event: Looked away for {completed_off_screen_duration:.1f} seconds.")
            log_cheating_event(f"OffScreen_{completed_off_screen_duration:.1f}s", frame, current_timestamp)

        # 5. Data Logging (CSV)
        write_csv_log_row(
            timestamp=current_timestamp,
            frame_count=frame_count,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            phone_detected_flag=phone_detected_flag,
            phone_detected_raw=len(phone_bboxes_this_frame) > 0, # Raw detection this frame
            is_anomaly=anomaly_detected,
            anomaly_score=anomaly_score if anomaly_detector.is_trained else np.nan,
            anomaly_features=current_features,
            current_off_screen_duration=current_off_screen_duration,
            completed_off_screen_duration=completed_off_screen_duration,
            face_bbox=face_bbox,
            landmarks_xy=landmarks_xy
        )

        # 6. Prepare Data for Visualization Plots (if enabled)
        if settings.ENABLE_INTERACTIVE_PLOTS and vis_data is not None:
            current_time_relative = (current_timestamp - start_time_monitoring).total_seconds()
            vis_data['time'].append(current_time_relative)
            vis_data['yaw'].append(yaw_deg if yaw_deg is not None else np.nan)
            vis_data['pitch'].append(pitch_deg if pitch_deg is not None else np.nan)
            vis_data['phone_detected'].append(float(phone_detected_flag)) # Use flag for plot

            # Append landmark data (or NaN if not detected/available)
            for i in range(settings.NUM_LANDMARKS):
                x_val = landmarks_xy[i][0] if landmarks_xy and len(landmarks_xy) > i else np.nan
                y_val = landmarks_xy[i][1] if landmarks_xy and len(landmarks_xy) > i else np.nan
                vis_data[f'lm{i}_x'].append(x_val)
                vis_data[f'lm{i}_y'].append(y_val)

        # 7. Calculate FPS
        now = time.time()
        elapsed_fps = now - last_fps_time
        fps = 0
        if elapsed_fps > 1.0: # Update FPS roughly every second
            fps = accumulated_frames_fps / elapsed_fps
            last_fps_time = now
            accumulated_frames_fps = 0

        # 8. Render Overlays onto Frame (conditionally based on config)
        display_frame = render_overlays(
            display_frame=display_frame,
            fps=fps,
            gaze_results=gaze_results, # Pass the results object
            phone_bboxes=phone_bboxes_this_frame,
            phone_detected_flag=phone_detected_flag,
            gaze_duration_tracker=gaze_duration_tracker,
            anomaly_detector=anomaly_detector,
            anomaly_detected=anomaly_detected,
            anomaly_score=anomaly_score
        )

        # 9. Update Interactive Plots (if enabled)
        if settings.ENABLE_INTERACTIVE_PLOTS and vis_objects is not None:
             update_visualization(frame_count, gaze_duration_tracker, anomaly_detector)

        # 10. Display Frame
        cv2.imshow(settings.WINDOW_NAME_MONITOR, display_frame)

        # 11. Check for Quit Key
        key = cv2.waitKey(1) & 0xFF # Wait 1ms
        if key == ord('q'):
            print("'q' pressed, exiting monitoring loop.")
            break

    # --- End of Loop ---


def cleanup(cap):
    """Releases resources."""
    print("\n--- Cleaning Up Resources ---")
    if cap:
        cap.release()
        print("Webcam released.")
    close_csv_logging() # Close CSV file
    if settings.ENABLE_INTERACTIVE_PLOTS:
        close_plots() # Close matplotlib window
    cv2.destroyAllWindows()
    print("OpenCV windows closed.")
    print("--- Cleanup Complete ---")


if __name__ == "__main__":
    # Ensure model directory structure might be needed (though checked in init)
    if not settings.L2CS_MODEL_DIR.parent.exists():
         print(f"Warning: Base 'models' directory not found at {settings.L2CS_MODEL_DIR.parent}")
         print("Model loading might fail if paths are incorrect.")

    # 1. Initialize
    phone_detector, gaze_pipeline = initialize_models()
    cap = initialize_camera()
    anomaly_detector = GazeAnomalyDetector()
    # Default tracker initially, thresholds updated after calibration
    gaze_duration_tracker = GazeDurationTracker()
	
    # 2. Calibration
    calibration_features, off_screen_thresholds = enhanced_calibration_mode(
        gaze_pipeline, cap, anomaly_detector # Pass the instance
    )

    # 3. Train Anomaly Detector (if calibration data exists)
    anomaly_training_successful = False
    if calibration_features:
        anomaly_training_successful = anomaly_detector.train(calibration_features)
    else:
        print("Calibration aborted or failed, skipping anomaly detector training.")


    if not anomaly_training_successful:
        print("*****************************************************")
        print("Anomaly detector training FAILED or insufficient data.")
        print(">>> Anomaly detection will be DISABLED. <<<")
        print("*****************************************************")

    # 4. Set Gaze Tracker Thresholds (use calibrated or default)
    if off_screen_thresholds:
        gaze_duration_tracker.set_thresholds(
            off_screen_thresholds['yaw'], off_screen_thresholds['pitch']
        )
    else:
        print("Calibration did not provide thresholds, using defaults for off-screen tracker.")
        # Tracker already initialized with defaults, just print confirmation
        print(f"Using default thresholds: Yaw={gaze_duration_tracker.yaw_threshold:.1f}, Pitch={gaze_duration_tracker.pitch_threshold:.1f}")


    # 5. Setup Logging and Visualization
    csv_logging_ok = setup_csv_logging()
    if not csv_logging_ok:
         print("Warning: CSV Logging failed to initialize.")
    # Setup plots (will be None if disabled in config)
    vis_objects, vis_data = setup_visualization()


    # 6. Run Main Loop
    try:
        main_loop(cap, phone_detector, gaze_pipeline, anomaly_detector, gaze_duration_tracker, vis_objects, vis_data)
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")
    except Exception as e:
         print(f"\n--- UNEXPECTED ERROR IN MAIN LOOP ---")
         print(f"Error: {e}")
         import traceback
         traceback.print_exc() # Print detailed traceback
         print(f"-------------------------------------")
    finally:
        # 7. Cleanup
        cleanup(cap)

    print("Application finished.")