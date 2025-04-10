import cv2
import time
import numpy as np
import datetime # Need this for timestamping potential errors during calibration

# Import necessary components and config
from config import settings
# Need GazeAnomalyDetector only for its _calculate_features method during calibration
from anomaly_detection.detector import GazeAnomalyDetector


def enhanced_calibration_mode(gaze_pipeline, cap, anomaly_detector_instance):
    """
    Runs an interactive calibration process to collect gaze data and determine thresholds.

    Args:
        gaze_pipeline: The initialized L2CS Pipeline object.
        cap: The OpenCV VideoCapture object.
        anomaly_detector_instance: An instance of GazeAnomalyDetector to use its feature calculation.


    Returns:
        tuple: (list_of_calibration_features, dict_of_thresholds)
               Returns (None, None) if calibration fails or is aborted.
               thresholds dict contains {'yaw': float, 'pitch': float}.
    """
    print("\n--- Starting Enhanced Calibration (Fullscreen) ---")
    print(f"Please follow the instructions. Look at the markers when they appear.")
    print(f"Calibration requires stable face detection. Press 'q' at any time to abort.")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_w <= 0 or frame_h <= 0:
        print("Error: Invalid frame dimensions from camera during calibration.")
        return None, None
    print(f"Calibration using frame dimensions: {frame_w}x{frame_h}")

    calibration_features_list = []
    calibration_raw_yaw = []
    calibration_raw_pitch = []
    calibration_landmarks_list = [] # Store landmarks during calibration too

    # Use targets from config
    targets = settings.CALIBRATION_TARGETS

    cv2.namedWindow(settings.WINDOW_NAME_CALIBRATION, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(settings.WINDOW_NAME_CALIBRATION, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print("Calibration window set to fullscreen. Press 'q' to exit.")

    try: # Wrap the main calibration loop in try...finally for cleanup
        for target_name, (tx_rel, ty_rel) in targets.items():
            tx_px = int(tx_rel * frame_w)
            ty_px = int(ty_rel * frame_h)

            print(f"Calibration step: Look at the '{target_name}' marker.")
            start_time = time.time()
            samples_this_target = 0

            while time.time() - start_time < settings.CALIBRATION_DURATION_PER_TARGET:
                ret, frame = cap.read()
                timestamp = datetime.datetime.now() # For error logging if needed
                if not ret or frame is None:
                    print(f"Warning: Failed to read frame during calibration ({target_name}) at {timestamp}")
                    time.sleep(0.05) # Wait a bit longer if frames fail
                    continue

                frame = cv2.flip(frame, 1)
                display_frame = frame.copy() # Work on a copy

                # Draw target marker
                cv2.circle(display_frame, (tx_px, ty_px), settings.CALIBRATION_TARGET_RADIUS, settings.CALIBRATION_TARGET_COLOR, -1)
                # Draw instructions
                cv2.putText(display_frame, f"Look at: {target_name}", (frame_w // 2 - 150, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, settings.CALIBRATION_TEXT_COLOR, 2)
                remaining = settings.CALIBRATION_DURATION_PER_TARGET - (time.time() - start_time)
                cv2.putText(display_frame, f"Time left: {remaining:.1f}s", (frame_w // 2 - 100, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, settings.CALIBRATION_TEXT_COLOR, 2)
                cv2.putText(display_frame, f"Samples: {len(calibration_features_list)}", (frame_w // 2 - 100, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, settings.CALIBRATION_TEXT_COLOR, 2)


                current_yaw, current_pitch = None, None
                landmarks_flat = None
                try:
                    # Gaze estimation step
                    results = gaze_pipeline.step(frame) # Use original frame for detection

                    if results and results.pitch is not None and len(results.pitch) > 0:
                        current_yaw = float(results.yaw[0])
                        current_pitch = float(results.pitch[0])
                        landmarks_np = results.landmarks[0] if results.landmarks is not None and len(results.landmarks) > 0 else None

                        if landmarks_np is not None and landmarks_np.shape == (settings.NUM_LANDMARKS, 2):
                            landmarks_flat = landmarks_np.flatten().tolist()
                            calibration_landmarks_list.append(landmarks_flat) # Add landmarks for features
                        else:
                            landmarks_flat = [0.0] * (settings.NUM_LANDMARKS * 2) # Use zeros if landmarks incorrect
                            calibration_landmarks_list.append(landmarks_flat)


                        calibration_raw_yaw.append(current_yaw)
                        calibration_raw_pitch.append(current_pitch)

                        features = anomaly_detector_instance._calculate_features(current_yaw, current_pitch, False,landmarks_flat if settings.INCLUDE_LANDMARKS_IN_FEATURES else None)
                        calibration_features_list.append(features)
                        samples_this_target += 1

                        # Display current gaze (optional, can be noisy)
                        cv2.putText(display_frame, f"Yaw: {current_yaw:.1f}", (20, frame_h - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, settings.CALIBRATION_INFO_COLOR, 2)
                        cv2.putText(display_frame, f"Pitch: {current_pitch:.1f}", (20, frame_h - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, settings.CALIBRATION_INFO_COLOR, 2)
                    else:
                         # No face detected or gaze estimation failed for this frame
                        cv2.putText(display_frame, "Gaze Not Detected", (20, frame_h - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # Red text

                except Exception as e:
                    # Catch potential errors during the gaze step itself
                    print(f"Error during calibration gaze step for {target_name} at {timestamp}: {e}")
                    cv2.putText(display_frame, "Gaze Processing Error", (20, frame_h - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # Red text


                cv2.imshow(settings.WINDOW_NAME_CALIBRATION, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Calibration aborted by user.")
                    # No need to destroy window here, finally block handles it
                    return None, None # Signal abortion

            print(f"  Collected {samples_this_target} valid gaze samples for {target_name}.")

        # --- Calibration loop finished ---
        print(f"Collected a total of {len(calibration_features_list)} data points.")

        # Show completion message
        ret, frame = cap.read() # Get one last frame
        if ret and frame is not None:
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "Calibration Complete!", (frame_w // 2 - 200, frame_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3) # Green text
            cv2.imshow(settings.WINDOW_NAME_CALIBRATION, frame)
            cv2.waitKey(2000) # Display for 2 seconds

    except Exception as e:
         print(f"An unexpected error occurred during the calibration loop: {e}")
         return None, None # Signal failure
    finally:
        cv2.destroyWindow(settings.WINDOW_NAME_CALIBRATION)
        print("--- Calibration Data Collection Finished ---")


    # --- Process collected data ---
    thresholds = {'yaw': settings.DEFAULT_YAW_THRESHOLD, 'pitch': settings.DEFAULT_PITCH_THRESHOLD} # Start with defaults

    if len(calibration_features_list) < settings.MIN_CALIBRATION_SAMPLES:
        print(f"Warning: Only collected {len(calibration_features_list)} valid samples (Min required: {settings.MIN_CALIBRATION_SAMPLES}).")
        print("Anomaly detection training may fail or be unreliable.")
        print("Using default off-screen thresholds.")
        # Keep features list for potential training attempt, but return default thresholds
        return calibration_features_list, thresholds
    else:
        print(f"Proceeding to calculate off-screen thresholds based on {len(calibration_raw_yaw)} raw gaze points.")
        if not calibration_raw_yaw or not calibration_raw_pitch:
            print("Warning: No raw gaze data collected, despite having feature samples. Using default thresholds.")
        else:
            try:
                # Use percentiles to filter outliers before calculating thresholds
                yaw_q_low, yaw_q_high = np.percentile(calibration_raw_yaw, [settings.CALIBRATION_PERCENTILE_LOW, settings.CALIBRATION_PERCENTILE_HIGH])
                pitch_q_low, pitch_q_high = np.percentile(calibration_raw_pitch, [settings.CALIBRATION_PERCENTILE_LOW, settings.CALIBRATION_PERCENTILE_HIGH])

                # Filter the raw data
                filtered_yaw = [y for y in calibration_raw_yaw if yaw_q_low <= y <= yaw_q_high]
                filtered_pitch = [p for p in calibration_raw_pitch if pitch_q_low <= p <= pitch_q_high]

                if not filtered_yaw or not filtered_pitch:
                    print("Warning: Could not determine gaze range after filtering outliers. Using defaults.")
                else:
                    # Calculate thresholds based on the max absolute filtered values plus a buffer
                    max_abs_yaw = np.max(np.abs(filtered_yaw))
                    max_abs_pitch = np.max(np.abs(filtered_pitch))

                    # Add buffer from config
                    yaw_threshold = max_abs_yaw + settings.CALIBRATION_YAW_BUFFER
                    pitch_threshold = max_abs_pitch + settings.CALIBRATION_PITCH_BUFFER

                    print("Calibration-based Off-Screen Thresholds Calculated:")
                    print(f"  Filtered Yaw Range ({settings.CALIBRATION_PERCENTILE_LOW}-{settings.CALIBRATION_PERCENTILE_HIGH}%): [{np.min(filtered_yaw):.1f}, {np.max(filtered_yaw):.1f}]")
                    print(f"  Filtered Pitch Range ({settings.CALIBRATION_PERCENTILE_LOW}-{settings.CALIBRATION_PERCENTILE_HIGH}%): [{np.min(filtered_pitch):.1f}, {np.max(filtered_pitch):.1f}]")
                    print(f"  ==> Calculated Thresholds -> Yaw: {yaw_threshold:.2f}, Pitch: {pitch_threshold:.2f}")
                    thresholds = {'yaw': yaw_threshold, 'pitch': pitch_threshold}

            except Exception as e:
                print(f"Error calculating thresholds from calibration data: {e}. Using defaults.")
                # Keep default thresholds if calculation fails

    return calibration_features_list, thresholds