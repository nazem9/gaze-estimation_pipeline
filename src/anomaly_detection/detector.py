import numpy as np
from collections import deque
from sklearn.ensemble import IsolationForest

# Import constants from config
from config import settings

class GazeAnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(
            contamination=settings.IFOREST_CONTAMINATION,
            random_state=settings.IFOREST_RANDOM_STATE,
            n_estimators=settings.IFOREST_N_ESTIMATORS,
            max_samples=settings.IFOREST_MAX_SAMPLES
        )
        # Feature buffer no longer needed here if calibration handles it
        self.is_trained = False
        self.anomaly_scores = deque(maxlen=settings.PLOT_HISTORY_LENGTH)
        self.current_features = [] # Store latest features for logging/prediction

    def _calculate_features(self, yaw, pitch, phone_detected, landmarks_flat):
        """Calculates features for anomaly detection."""
        _yaw = yaw if yaw is not None else 0.0
        _pitch = pitch if pitch is not None else 0.0
        _landmarks = landmarks_flat if landmarks_flat is not None else [0.0] * (settings.NUM_LANDMARKS * 2)
        
        # Ensure landmarks list has the correct length, padding with 0 if needed
        if len(_landmarks) < settings.NUM_LANDMARKS * 2:
            _landmarks.extend([0.0] * (settings.NUM_LANDMARKS * 2 - len(_landmarks)))
        elif len(_landmarks) > settings.NUM_LANDMARKS * 2:
             _landmarks = _landmarks[:settings.NUM_LANDMARKS * 2] # Truncate if too long


        feature_prod = _yaw * _pitch
        feature_diff = _yaw - _pitch
        features = [_yaw, _pitch, feature_prod, feature_diff, float(phone_detected)] + _landmarks
        return features

    def prepare_features_for_prediction(self, yaw, pitch, phone_detected, landmarks_flat):
        """Calculates and stores the latest features."""
        self.current_features = self._calculate_features(yaw, pitch, phone_detected, landmarks_flat)
        return self.current_features

    def train(self, calibration_features):
        """Trains the Isolation Forest model on calibration data."""
        if not calibration_features or len(calibration_features) < settings.MIN_CALIBRATION_SAMPLES:
            print(f"Error: Not enough calibration data ({len(calibration_features)} samples) "
                  f"to train anomaly detector. Need at least {settings.MIN_CALIBRATION_SAMPLES}.")
            self.is_trained = False
            return False

        X = np.array(calibration_features)

        # Check for NaN/Inf more robustly
        if not np.all(np.isfinite(X)):
            print(f"Warning: Non-finite values (NaN or Inf) found in {np.sum(~np.isfinite(X))} entries "
                  "of calibration data. Skipping training.")
            self.is_trained = False
            return False
        if X.shape[0] < 2: # Need at least 2 samples
             print("Warning: Less than 2 valid samples for training. Skipping.")
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
            print("This might happen if all calibration features are identical or insufficient samples.")
            self.is_trained = False
            return False
        except Exception as e:
            print(f"Error during Isolation Forest training: {e}")
            self.is_trained = False
            return False

    def predict(self):
        """Predicts anomaly based on the latest stored features."""
        if not self.is_trained:
            self.anomaly_scores.append(0.0) # Append neutral score if not trained
            return False, 0.0

        if not self.current_features:
            print("Warning: No features available for prediction.")
            self.anomaly_scores.append(0.0)
            return False, 0.0

        features_arr = np.array([self.current_features])

        if not np.all(np.isfinite(features_arr)):
            print("Warning: NaN or Inf found in features for prediction. Returning non-anomalous.")
            self.anomaly_scores.append(0.0) # Append neutral score
            return False, 0.0

        try:
            # Ensure model is fitted before predicting
            if not hasattr(self.model, "estimators_"):
                 print("Error: Isolation Forest model is not fitted. Cannot predict.")
                 self.anomaly_scores.append(0.0)
                 return False, 0.0

            prediction = self.model.predict(features_arr)
            # Use decision_function for a continuous score (lower is more anomalous)
            anomaly_score = self.model.decision_function(features_arr)[0]
            self.anomaly_scores.append(anomaly_score)
            is_anomaly = (prediction[0] == -1) # -1 indicates anomaly in IsolationForest
            return is_anomaly, anomaly_score
        except Exception as e:
            print(f"Error during Isolation Forest prediction: {e}")
            self.anomaly_scores.append(0.0) # Append neutral score on error
            return False, 0.0