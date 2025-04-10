from ultralytics import YOLO
import cv2
import time
import numpy as np
# Import config
from config import settings

class PhoneDetector:
    def __init__(self, model_path=settings.YOLO_MODEL_PATH, device=settings.DEVICE):
        self.model_path = model_path
        self.device = device
        self.model = self._load_model()
        self.consecutive_frames = 0
        self.phone_detected_flag = False
        self.last_bboxes = [] # Store last detected bounding boxes

    def _load_model(self):
        try:
            print(f"Loading YOLO model from: {self.model_path} for device: {self.device}")
            model = YOLO(self.model_path)
            # Perform a dummy inference to potentially speed up the first real one
            dummy_img = np.zeros((640,640,3),np.uint8)
            _ = model.predict(dummy_img , verbose=False, device=self.device)
            print("YOLO model loaded successfully.")
            return model
        except Exception as e:
            print(f"FATAL ERROR loading YOLO model: {e}. Ensure '{self.model_path}' is available.")
            print("Phone detection will be disabled.")
            return None

    def detect(self, frame):
        """
        Detects phones in the frame.

        Args:
            frame: The input image frame (NumPy array).

        Returns:
            tuple: (bool: phone_detected_flag, list: list_of_bboxes)
                   bboxes are in [x1, y1, x2, y2] format.
        """
        if self.model is None:
            self.last_bboxes = []
            return self.phone_detected_flag, self.last_bboxes, False

        phone_bboxes_this_frame = []
        try:
            results = self.model.predict(
                frame,
                classes=[settings.PHONE_CLASS_ID],
                verbose=False,
                conf=settings.PHONE_CONFIDENCE_THRESHOLD,
                device=self.device
            )

            if results and len(results) > 0:
                res = results[0] # First result object
                if hasattr(res, 'boxes') and res.boxes is not None and len(res.boxes) > 0:
                    # Iterate through detected boxes of the target class
                    cpu_boxes = res.boxes.cpu() # Move results to CPU for processing
                    for i in range(len(cpu_boxes.cls)):
                        # No need to check class ID again if filtered in predict, but double-check is safe
                        if int(cpu_boxes.cls[i]) == settings.PHONE_CLASS_ID:
                           box = cpu_boxes.xyxy[i].numpy().astype(int) # [x1, y1, x2, y2]
                           phone_bboxes_this_frame.append(box.tolist())

        except Exception as e:
            print(f"Error during YOLO prediction: {e}")
            # Reset detection state on error? Maybe not, could be transient.
            phone_bboxes_this_frame = []


        # Update consecutive frame count and flag
        phone_detected_this_frame = len(phone_bboxes_this_frame) > 0

        if phone_detected_this_frame:
            self.consecutive_frames += 1
        else:
            self.consecutive_frames = 0 # Reset if no phone detected

        # Check if the threshold for consecutive frames is met
        new_flag_state = self.consecutive_frames >= settings.PHONE_DETECTION_CONSECUTIVE_FRAMES

        # Check if the flag just turned True
        triggered_now = new_flag_state and not self.phone_detected_flag

        self.phone_detected_flag = new_flag_state
        self.last_bboxes = phone_bboxes_this_frame # Store boxes from this frame

        # Return the current flag state, the raw boxes from *this* frame,
        # and whether the flag was just triggered *now*
        return self.phone_detected_flag, self.last_bboxes, triggered_now