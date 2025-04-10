import torch
from pathlib import Path

# --- Core Model & Device Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
YOLO_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / 'yolo11l.pt' # Or specify a full path
L2CS_MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "L2CSNet" / "Gaze360"
L2CS_MODEL_NAME = "L2CSNet_gaze360.pkl"
L2CS_ARCH = 'ResNet50'

# --- Detection Thresholds & Parameters ---
PHONE_CLASS_ID = 67
PHONE_CONFIDENCE_THRESHOLD = 0.30
PHONE_DETECTION_CONSECUTIVE_FRAMES = 5 # Renamed for clarity

# --- Anomaly Detection Parameters ---
IFOREST_CONTAMINATION = 0.05
IFOREST_N_ESTIMATORS = 150
IFOREST_MAX_SAMPLES = 'auto'
IFOREST_RANDOM_STATE = 42
MIN_CALIBRATION_SAMPLES = 100
INCLUDE_LANDMARKS_IN_FEATURES = False
# --- Gaze Tracking Parameters ---
DEFAULT_YAW_THRESHOLD = 25.0
DEFAULT_PITCH_THRESHOLD = 25.0
CALIBRATION_YAW_BUFFER = 5.0    # Degrees to add to max calibration yaw
CALIBRATION_PITCH_BUFFER = 8.0  # Degrees to add to max calibration pitch
CALIBRATION_PERCENTILE_LOW = 5  # Lower percentile for filtering calibration gaze
CALIBRATION_PERCENTILE_HIGH = 95 # Upper percentile for filtering calibration gaze
CALIBRATION_DURATION_PER_TARGET = 5 # Seconds per calibration point

# --- Visualization & Logging Parameters ---
PLOT_HISTORY_LENGTH = 200
VISUALIZATION_UPDATE_INTERVAL = 5 # Update plots every N frames
OFF_SCREEN_PLOT_BAR_COUNT = 15
OFF_SCREEN_DURATION_THRESHOLD = 2.5 # Minimum duration to log/plot an off-screen event
NUM_LANDMARKS = 5 # Expected number of landmarks from L2CS
LOG_FOLDER_EVENTS = "cheating_logs"
LOG_FOLDER_DATA = "logs"
VIZUALIZATION_PATH = Path(__file__).parent.parent / "cheating_logs" / "visualizations"
# --- NEW CONFIGURATION FLAGS ---
# Decide whether to show interactive matplotlib plots
ENABLE_INTERACTIVE_PLOTS = True
# Decide whether to draw landmarks on the OpenCV video frame
RENDER_LANDMARKS_ON_FRAME = True
# Decide whether to draw the gaze vector on the OpenCV video frame
RENDER_GAZE_VECTOR_ON_FRAME = True
# Decide whether to draw info text (FPS, gaze angles, status) on the OpenCV video frame
RENDER_INFO_TEXT_ON_FRAME = True
# Decide whether to draw bounding boxes for detected phones
RENDER_PHONE_BOX_ON_FRAME = True
# Decide whether to draw the face bounding box from L2CS
RENDER_FACE_BOX_ON_FRAME = True

# --- Calibration Targets ---
CALIBRATION_TARGETS = {
    "Center": (0.5, 0.5), "Top-Left": (0.1, 0.1), "Top-Right": (0.9, 0.1),
    "Bottom-Left": (0.1, 0.9), "Bottom-Right": (0.9, 0.9), "Mid-Left": (0.05, 0.5),
    "Mid-Right": (0.95, 0.5), "Mid-Top": (0.5, 0.05), "Mid-Bottom": (0.5, 0.95),
}
CALIBRATION_TARGET_RADIUS = 25
CALIBRATION_TARGET_COLOR = (0, 255, 255) # Yellow
CALIBRATION_TEXT_COLOR = (255, 255, 255) # White
CALIBRATION_INFO_COLOR = (50, 200, 250)  # Light Blue/Orange
# --- Frame Display ---
WINDOW_NAME_CALIBRATION = "Calibration"
WINDOW_NAME_MONITOR = "Cheating Detection Monitor"