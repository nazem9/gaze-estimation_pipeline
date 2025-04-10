import cv2
import torch
import time
import datetime
import csv
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm # For progress bar

# --- Import L2CS Pipeline and relevant settings ---
# Assume this script is run from the 'cheating_detection' root directory,
# or adjust sys.path if needed.
try:
    from src.config import settings # Import shared settings
    from l2cs import Pipeline
except ImportError:
    import sys
    # Add the 'src' directory to the path if running from 'tools/' directory
    project_root = Path(__file__).parent.parent
    src_path = project_root / "src"
    if src_path.is_dir():
        sys.path.insert(0, str(project_root))
        print(f"Added {project_root} to sys.path")
        from src.config import settings
        from l2cs import Pipeline
    else:
        print("Error: Cannot find 'src' directory. Please run from the project root "
              "or ensure 'src' is in the Python path.")
        sys.exit(1)

# --- Constants ---
NUM_LANDMARKS = settings.NUM_LANDMARKS # Use from central config

def format_value_for_csv(val, precision=4):
    """Helper to format numbers for CSV, handling None/NaN -> empty string."""
    if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
        return "" # Represent missing values as empty strings
    if isinstance(val, (float, np.floating)):
        return f"{val:.{precision}f}"
    if isinstance(val, (int, np.integer)):
        return str(val)
    return str(val)

def collect_data_from_video(video_path: Path, output_csv_path: Path, model_path: Path, arch: str, device: torch.device):
    """
    Processes a video file to extract gaze and landmark data into a CSV file.

    Args:
        video_path (Path): Path to the input video file.
        output_csv_path (Path): Path to save the output CSV file.
        model_path (Path): Path to the L2CS model weights file.
        arch (str): Model architecture (e.g., 'ResNet50').
        device (torch.device): Device to run the model on (CPU or CUDA).
    """
    print("--- Starting Data Collection ---")
    print(f"Input Video: {video_path}")
    print(f"Output CSV: {output_csv_path}")
    print(f"L2CS Model: {model_path}")
    print(f"Architecture: {arch}")
    print(f"Device: {device}")

    # --- Validate Inputs ---
    if not video_path.is_file():
        print(f"Error: Video file not found at {video_path}")
        return
    if not model_path.is_file():
        print(f"Error: L2CS model file not found at {model_path}")
        return

    # Create output directory if it doesn't exist
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Initialize Models ---
    try:
        gaze_pipeline = Pipeline(
            weights=model_path,
            arch=arch,
            device=device
        )
        print("L2CS Gaze Pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading L2CS Gaze Pipeline: {e}")
        return

    # --- Initialize Video Capture ---
    cap = cv2.VideoCapture(str(video_path),)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Warning: Could not determine video FPS. Timestamps might be inaccurate.")
        fps = 30 # Assume default if unknown
    print(f"Video Properties: {frame_count_total} frames, {fps:.2f} FPS")


    # --- Prepare CSV Output ---
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)

            # Write Header
            header = ["Timestamp_ms", "Frame", "Yaw", "Pitch"]
            for i in range(NUM_LANDMARKS):
                header.extend([f"LM_{i}_X", f"LM_{i}_Y"])
            csv_writer.writerow(header)

            print("\n--- Processing Video Frames ---")
            # --- Processing Loop ---
            frame_num = 0
            # Use tqdm for a progress bar
            with tqdm(total=frame_count_total, unit="frame", desc="Processing") as pbar:
                while True:
                    success, frame = cap.read()
                    if not success:
                        break # End of video

                    frame_num += 1
                    pbar.update(1) # Update progress bar

                    # Calculate timestamp (milliseconds from start of video)
                    # Note: CAP_PROP_POS_MSEC can be unreliable; calculating is often more consistent
                    timestamp_ms = (frame_num - 1) * (1000.0 / fps) if fps > 0 else (frame_num - 1) * 33.33

                    # Initialize results for this frame
                    yaw_deg, pitch_deg = np.nan, np.nan
                    landmarks_flat = [np.nan] * (NUM_LANDMARKS * 2)

                    try:
                        # Run gaze estimation
                        # Note: L2CS expects BGR format from OpenCV
                        gaze_results = gaze_pipeline.step(frame)

                        if gaze_results and gaze_results.pitch is not None and len(gaze_results.pitch) > 0:
                            yaw_deg = float(gaze_results.yaw[0])
                            pitch_deg = float(gaze_results.pitch[0])

                            # Extract landmarks if available and correctly shaped
                            if gaze_results.landmarks is not None and len(gaze_results.landmarks) > 0:
                                lm_np = gaze_results.landmarks[0]
                                if lm_np.shape == (NUM_LANDMARKS, 2):
                                    landmarks_flat = lm_np.flatten().tolist()

                    except ValueError:
                        # L2CS raises ValueError if no face is detected - expected behavior
                        pass # Keep yaw/pitch/landmarks as NaN
                    except Exception as e:
                        print(f"\nWarning: Error during gaze estimation on frame {frame_num}: {e}")
                        # Keep results as NaN for this frame, but continue processing

                    # --- Write Row to CSV ---
                    row_data = [
                        format_value_for_csv(timestamp_ms, precision=2), # Milliseconds with 2 decimal places
                        frame_num,
                        format_value_for_csv(yaw_deg),
                        format_value_for_csv(pitch_deg)
                    ]
                    row_data.extend(map(format_value_for_csv, landmarks_flat))
                    csv_writer.writerow(row_data)

            # --- End of Loop ---
            print(f"\n--- Processing Complete ---")
            print(f"Successfully processed {frame_num} frames.")

    except IOError as e:
        print(f"Error: Could not write to CSV file {output_csv_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
    finally:
        # --- Cleanup ---
        if cap.isOpened():
            cap.release()
            print("Video capture released.")
        print(f"Data saved to {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract gaze and landmark data from a video file.")

    parser.add_argument("--video", type=str, required=True,
                        help="Path to the input video file.")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to the output CSV file. "
                             "(Default: <video_filename>_gaze_data.csv in ./logs/)")
    parser.add_argument("--model", type=str,
                        default=str(settings.L2CS_MODEL_DIR / settings.L2CS_MODEL_NAME),
                        help=f"Path to the L2CS model weights file. "
                             f"(Default: {settings.L2CS_MODEL_DIR / settings.L2CS_MODEL_NAME})")
    parser.add_argument("--arch", type=str, default=settings.L2CS_ARCH,
                        help=f"Model architecture (e.g., 'ResNet50', 'ResNet18'). "
                             f"(Default: {settings.L2CS_ARCH})")
    parser.add_argument("--device", type=str, default=str(settings.DEVICE),
                        help="Device to use ('cuda' or 'cpu'). "
                             f"(Default: {settings.DEVICE})")

    args = parser.parse_args()

    # --- Prepare Paths and Device ---
    video_path = Path(args.video).resolve() # Get absolute path

    # Determine output path
    if args.output:
        output_csv_path = Path(args.output).resolve()
    else:
        # Default output: place in 'logs' subdir relative to project root,
        # based on video filename
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        output_csv_path = log_dir / f"{video_path.stem}_gaze_data.csv"

    model_path = Path(args.model).resolve()
    device = torch.device(args.device)

    # --- Run Data Collection ---
    start_time = time.time()
    collect_data_from_video(video_path, output_csv_path, model_path, args.arch, device)
    end_time = time.time()

    print(f"Total collection time: {end_time - start_time:.2f} seconds.")