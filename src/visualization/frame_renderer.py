import cv2
import numpy as np
from l2cs import render # Use the original render for gaze vector if needed

# Import config and potentially other components if needed for status text
from config import settings

def render_overlays(
    display_frame,
    fps,
    gaze_results, # The result object from l2cs Pipeline.step()
    phone_bboxes, # List of [x1, y1, x2, y2] for phones
    phone_detected_flag,
    gaze_duration_tracker, # Instance of GazeDurationTracker
    anomaly_detector, # Instance of GazeAnomalyDetector
    anomaly_detected, # Boolean result from detector.predict()
    anomaly_score     # Score result from detector.predict()
    ):
    """
    Draws all requested visualizations onto the OpenCV frame.

    Args:
        display_frame: The OpenCV frame (NumPy array) to draw on (will be modified).
        fps: Current frames per second.
        gaze_results: L2CS result object. Can be None.
        phone_bboxes: List of bounding boxes for detected phones.
        phone_detected_flag: Boolean indicating if phone presence threshold is met.
        gaze_duration_tracker: The GazeDurationTracker instance.
        anomaly_detector: The GazeAnomalyDetector instance.
        anomaly_detected: Boolean result of anomaly prediction.
        anomaly_score: Score from anomaly prediction.

    Returns:
        The modified display_frame.
    """
    height, width, _ = display_frame.shape
    # --- Render Elements based on Config ---

    # 1. L2CS BBox, Gaze Vector, Landmarks (conditionally)
    if gaze_results:
        # Use a temporary copy if we only want parts of the original render
        temp_frame_for_l2cs = display_frame.copy()
        
        # Check individual flags before calling the original render or drawing manually
        should_render_l2cs = False
        landmarks_to_draw = None
        
        if settings.RENDER_GAZE_VECTOR_ON_FRAME:
            should_render_l2cs = True # Original render draws gaze vector well
        if settings.RENDER_FACE_BOX_ON_FRAME:
             should_render_l2cs = True # Original render draws face box

        # Only draw landmarks if flag is set AND landmarks exist
        if settings.RENDER_LANDMARKS_ON_FRAME and gaze_results.landmarks is not None and len(gaze_results.landmarks) > 0:
            landmarks_to_draw = gaze_results.landmarks[0] # Assuming single face
            if landmarks_to_draw is not None:
                for (x, y) in landmarks_to_draw:
                    cv2.circle(display_frame, (int(x), int(y)), 3, (0, 255, 0), -1) # Green dots
            else:
                 # If calling l2cs render, let it handle landmarks if they exist
                 pass # l2cs render handles landmarks internally if present in results


        # Call the original l2cs render function if needed for gaze or face box
        if should_render_l2cs:
             try:
                 # Important: Pass the *original* results object to render
                 display_frame = render(display_frame, gaze_results) 
             except Exception as e:
                 # print(f"Warning: Error calling l2cs.render: {e}") # Avoid spamming console
                 pass # Continue drawing other overlays even if l2cs render fails

        # If l2cs render wasn't called, but we need landmarks, draw them now
        if not should_render_l2cs and landmarks_to_draw is not None and settings.RENDER_LANDMARKS_ON_FRAME:
              for (x, y) in landmarks_to_draw:
                    cv2.circle(display_frame, (int(x), int(y)), 3, (0, 255, 0), -1) # Green dots


    # 2. Phone Bounding Boxes (conditionally)
    if settings.RENDER_PHONE_BOX_ON_FRAME and phone_bboxes:
        for (x1, y1, x2, y2) in phone_bboxes:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 165, 255), 2) # Orange box
            cv2.putText(display_frame, "Phone", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    # 3. Information Text (conditionally)
    if settings.RENDER_INFO_TEXT_ON_FRAME:
        y_offset = 30
        text_scale = 0.6
        text_thickness = 2
        text_font = cv2.FONT_HERSHEY_SIMPLEX

        # FPS
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (width - 100, 30),
                    text_font, 0.7, (0, 255, 0), text_thickness) # Green FPS

        # Gaze Angles
        yaw_deg, pitch_deg = None, None
        if gaze_results and gaze_results.pitch is not None and len(gaze_results.pitch) > 0:
             yaw_deg = float(gaze_results.yaw[0])
             pitch_deg = float(gaze_results.pitch[0])

        if yaw_deg is not None:
            cv2.putText(display_frame, f"Yaw: {yaw_deg:.1f}", (20, y_offset), text_font, text_scale, (255, 255, 0), text_thickness); y_offset += 25 # Cyan
            cv2.putText(display_frame, f"Pitch: {pitch_deg:.1f}", (20, y_offset), text_font, text_scale, (255, 255, 0), text_thickness); y_offset += 25
        else:
            cv2.putText(display_frame, "Gaze Not Detected", (20, y_offset), text_font, text_scale, (0, 0, 255), text_thickness); y_offset += 25 # Red

        # Anomaly Status
        if anomaly_detector.is_trained:
            status_text = "ANOMALY DETECTED" if anomaly_detected else "Normal"
            status_color = (0, 0, 255) if anomaly_detected else (0, 255, 0) # Red / Green
            cv2.putText(display_frame, f"Anomaly: {status_text} ({anomaly_score:.2f})", (20, y_offset),
                        text_font, text_scale, status_color, text_thickness); y_offset += 25
        else:
             cv2.putText(display_frame, "Anomaly Detection: INACTIVE", (20, y_offset),
                        text_font, text_scale, (0, 165, 255), text_thickness); y_offset += 25 # Orange

        # Off-Screen Status
        current_off_screen_duration = gaze_duration_tracker.get_current_off_screen_duration()
        last_completed_duration = gaze_duration_tracker.last_completed_duration

        if current_off_screen_duration > 0.1: # Show if currently off-screen
             off_screen_color = (50, 150, 255) # Orange-Red
             # Make color brighter/redder as duration increases? (Optional)
             # if current_off_screen_duration > settings.OFF_SCREEN_DURATION_THRESHOLD * 1.5:
             #    off_screen_color = (0, 0, 255) # Pure Red for long duration

             cv2.putText(display_frame, f"OFF-SCREEN: {current_off_screen_duration:.1f}s", (20, y_offset),
                        text_font, text_scale, off_screen_color, text_thickness); y_offset += 25
        elif last_completed_duration > 0: # Show briefly after returning
             # Only show return message if it was significant enough to be logged/plotted
             if last_completed_duration >= settings.OFF_SCREEN_DURATION_THRESHOLD:
                 cv2.putText(display_frame, f"Returned ({last_completed_duration:.1f}s off)", (20, y_offset),
                             text_font, text_scale, (0, 200, 100), text_thickness); y_offset += 25 # Green-Blue
             # else: # Optional: You could have a different message for short off-screen times
             #    cv2.putText(display_frame, f"Looked away briefly", (20, y_offset),
             #               text_font, text_scale, (200, 200, 0), text_thickness); y_offset += 25


        # Phone Status
        if phone_detected_flag:
             cv2.putText(display_frame, "PHONE DETECTED", (20, y_offset),
                        text_font, text_scale, (0, 165, 255), text_thickness); y_offset += 25 # Orange


    return display_frame