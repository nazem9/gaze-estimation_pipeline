# -----------------------------------------------------------------------------------
# Company: TensorSense
# Project: LaserGaze
# File: VisualizationOptions.py
# Description: A class to store visualization settings for rendering gaze vectors
#              on video frames.
# Author: Sergey Kuldin
# -----------------------------------------------------------------------------------


class VisualizationOptions:
    def __init__(self):
        self.color = (0, 255, 0)  # Green color for normal visualization
        self.warning_color = (0, 0, 255)  # Red color for warnings
        self.peripheral_color = (0, 255, 255)  # Yellow color for peripheral zone
        self.line_thickness = 2
        self.box_thickness = 1
        self.text_scale = 0.5