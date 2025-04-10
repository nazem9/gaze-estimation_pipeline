import datetime
from collections import deque
import numpy as np

# Import constants from config
from config import settings

class GazeDurationTracker:
    def __init__(self, yaw_threshold=settings.DEFAULT_YAW_THRESHOLD, pitch_threshold=settings.DEFAULT_PITCH_THRESHOLD):
        self.yaw_threshold = abs(yaw_threshold)
        self.pitch_threshold = abs(pitch_threshold)
        self.off_screen_start_time = None
        self.off_screen_durations = deque(maxlen=settings.OFF_SCREEN_PLOT_BAR_COUNT)
        self.last_completed_duration = 0.0 # Store the last completed off-screen duration

    def update(self, yaw, pitch):
        """
        Updates the off-screen duration based on current gaze angles.

        Returns:
            float: The duration of the *completed* off-screen event (0 if no event completed).
        """
        self.last_completed_duration = 0.0 # Reset completion flag

        # If face/gaze is lost, consider it potentially off-screen, but don't finalize duration yet
        if yaw is None or pitch is None:
            if self.off_screen_start_time is None:
                # Start timer if face is lost and wasn't already off-screen
                self.off_screen_start_time = datetime.datetime.now()
            # Don't complete the duration if face is lost; wait for it to reappear or continue
            return 0.0

        # Gaze is valid, check thresholds
        is_off_screen = abs(yaw) > self.yaw_threshold or abs(pitch) > self.pitch_threshold
        current_time = datetime.datetime.now()

        if is_off_screen:
            if self.off_screen_start_time is None:
                # Started looking off-screen
                self.off_screen_start_time = current_time
        else: # Is on-screen
            if self.off_screen_start_time is not None:
                # Just returned to the screen
                duration = (current_time - self.off_screen_start_time).total_seconds()
                if duration >= settings.OFF_SCREEN_DURATION_THRESHOLD: # Only record significant durations
                    self.off_screen_durations.append(duration)
                self.last_completed_duration = duration # Store this completed duration
                self.off_screen_start_time = None # Reset timer

        return self.last_completed_duration


    def get_current_off_screen_duration(self):
        """Returns the duration the user has been continuously looking off-screen, or 0."""
        if self.off_screen_start_time:
            return (datetime.datetime.now() - self.off_screen_start_time).total_seconds()
        return 0.0

    def set_thresholds(self, yaw_thresh, pitch_thresh):
        """Updates the thresholds used to determine if gaze is off-screen."""
        print(f"Updating off-screen thresholds: Yaw={abs(yaw_thresh):.2f}, Pitch={abs(pitch_thresh):.2f}")
        self.yaw_threshold = abs(yaw_thresh)
        self.pitch_threshold = abs(pitch_thresh)