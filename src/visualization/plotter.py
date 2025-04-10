import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from collections import deque

# Import config
from config import settings

# Global dictionary to hold visualization objects if plots are enabled
vis_objects = None
visualization_data = None


def setup_visualization():
    """Sets up the Matplotlib interactive plot figure and axes."""
    global vis_objects, visualization_data # Declare modification of globals

    if not settings.ENABLE_INTERACTIVE_PLOTS:
        print("Interactive plots disabled via config.")
        vis_objects = None
        visualization_data = None
        return None, None # Return None for both

    print("Setting up interactive plots...")
    plt.ion() # Interactive mode on
    fig = plt.figure(figsize=(16, 12)) # Keep size from original
    gs = GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3) # 4 rows for landmarks

    # --- Axes Definitions ---
    ax_gaze_ts = fig.add_subplot(gs[0, 0]) # Gaze angles time series
    ax_anomaly = fig.add_subplot(gs[1, 0]) # Anomaly score time series
    ax_phone = fig.add_subplot(gs[0, 1])   # Phone detection time series
    ax_offscreen = fig.add_subplot(gs[1, 1])# Off-screen duration bars
    ax_lm_x = fig.add_subplot(gs[2, 0])    # Landmark X coords time series
    ax_lm_y = fig.add_subplot(gs[2, 1])    # Landmark Y coords time series
    ax_gaze_scatter = fig.add_subplot(gs[3, :]) # Gaze distribution scatter

    # --- Line/Plot Objects ---
    lines = {}
    lines['yaw'], = ax_gaze_ts.plot([], [], label="Yaw", color="blue", lw=1)
    lines['pitch'], = ax_gaze_ts.plot([], [], label="Pitch", color="green", lw=1)

    lines['anomaly_score'], = ax_anomaly.plot([], [], label="Anomaly Score", color="red", lw=1)
    # Add the threshold line for anomaly score (approximate)
    ax_anomaly.axhline(0, color='grey', linestyle='--', lw=1, label='Approx. Threshold')

    lines['phone'], = ax_phone.plot([], [], label="Phone Detected", color="orange", lw=1.5, marker='.', linestyle='None')

    landmark_colors = plt.cm.viridis(np.linspace(0, 1, settings.NUM_LANDMARKS))
    for i in range(settings.NUM_LANDMARKS):
        lines[f'lm{i}_x'], = ax_lm_x.plot([], [], label=f"LM {i} X", color=landmark_colors[i], lw=1)
        lines[f'lm{i}_y'], = ax_lm_y.plot([], [], label=f"LM {i} Y", color=landmark_colors[i], lw=1)

    scatter_gaze = ax_gaze_scatter.scatter([], [], alpha=0.5, s=8, label="Gaze Points (Recent)")
    # Threshold lines for gaze scatter plot
    lines['yaw_thresh_pos'], = ax_gaze_scatter.plot([], [], 'r--', lw=1, label='Yaw Thresh')
    lines['yaw_thresh_neg'], = ax_gaze_scatter.plot([], [], 'r--', lw=1)
    lines['pitch_thresh_pos'], = ax_gaze_scatter.plot([], [], 'g--', lw=1, label='Pitch Thresh')
    lines['pitch_thresh_neg'], = ax_gaze_scatter.plot([], [], 'g--', lw=1)
    ax_gaze_scatter.axhline(0, color='grey', lw=0.5)
    ax_gaze_scatter.axvline(0, color='grey', lw=0.5)


    # --- Axes Styling ---
    ax_gaze_ts.set_title("Gaze Angles (Live)")
    ax_gaze_ts.set_ylabel("Degrees")
    ax_gaze_ts.legend(loc="upper left")
    ax_gaze_ts.grid(True, linestyle=':')

    ax_anomaly.set_title("Anomaly Score (Lower is more anomalous)")
    ax_anomaly.set_ylabel("Score")
    ax_anomaly.legend(loc="upper left")
    ax_anomaly.grid(True, linestyle=':')

    ax_phone.set_title("Phone Detection Status")
    ax_phone.set_ylabel("Detected (1) / Not Detected (0)")
    ax_phone.set_ylim(-0.1, 1.1)
    ax_phone.legend(loc="upper left")
    ax_phone.grid(True, linestyle=':')

    ax_offscreen.set_title(f"Recent Off-Screen Events (>{settings.OFF_SCREEN_DURATION_THRESHOLD:.1f}s)")
    ax_offscreen.set_ylabel("Duration (s)")
    ax_offscreen.set_xlabel("Event Index")
    ax_offscreen.grid(True, axis='y', linestyle=':')

    ax_lm_x.set_title("Landmark X Coordinates")
    ax_lm_x.set_ylabel("X Coordinate (pixels)")
    ax_lm_x.legend(loc="upper left", fontsize='small')
    ax_lm_x.grid(True, linestyle=':')

    ax_lm_y.set_title("Landmark Y Coordinates")
    ax_lm_y.set_ylabel("Y Coordinate (pixels)")
    ax_lm_y.legend(loc="upper left", fontsize='small')
    ax_lm_y.grid(True, linestyle=':')

    ax_gaze_scatter.set_title("Gaze Distribution (Yaw vs Pitch) & Thresholds")
    ax_gaze_scatter.set_xlabel("Yaw (degrees)")
    ax_gaze_scatter.set_ylabel("Pitch (degrees)")
    ax_gaze_scatter.grid(True, linestyle=':')
    ax_gaze_scatter.legend(loc='upper right')

    # Apply tight layout
    plt.tight_layout(pad=2.0)

    # Store objects
    vis_objects = {
        'fig': fig,
        'axes': { # Use names for easier access
            'gaze_ts': ax_gaze_ts, 'anomaly': ax_anomaly, 'phone': ax_phone,
            'offscreen': ax_offscreen, 'lm_x': ax_lm_x, 'lm_y': ax_lm_y,
            'gaze_scatter': ax_gaze_scatter
        },
        'lines': lines,
        'scatter': scatter_gaze,
        'bar_container': None # For the off-screen bars
    }

    # Setup data queues
    visualization_data = {
        'time': deque(maxlen=settings.PLOT_HISTORY_LENGTH),
        'yaw': deque(maxlen=settings.PLOT_HISTORY_LENGTH),
        'pitch': deque(maxlen=settings.PLOT_HISTORY_LENGTH),
        'phone_detected': deque(maxlen=settings.PLOT_HISTORY_LENGTH),
        # Anomaly scores are stored in the detector itself
    }
    # Add deques for landmark data
    for i in range(settings.NUM_LANDMARKS):
        visualization_data[f'lm{i}_x'] = deque(maxlen=settings.PLOT_HISTORY_LENGTH)
        visualization_data[f'lm{i}_y'] = deque(maxlen=settings.PLOT_HISTORY_LENGTH)


    print("Plot setup complete.")
    return vis_objects, visualization_data


def update_visualization(frame_count, gaze_duration_tracker, anomaly_detector):
    """Updates the Matplotlib plots with the latest data."""
    global vis_objects, visualization_data # Access globals

    # Exit if plots are not enabled or not set up
    if not settings.ENABLE_INTERACTIVE_PLOTS or vis_objects is None or visualization_data is None:
        return
    # Update only periodically
    if frame_count % settings.VISUALIZATION_UPDATE_INTERVAL != 0:
        return

    times = list(visualization_data['time'])
    if not times: # No data yet
        return

    # --- Update Time Series Plots ---
    vis_objects['lines']['yaw'].set_data(times, list(visualization_data['yaw']))
    vis_objects['lines']['pitch'].set_data(times, list(visualization_data['pitch']))
    # Anomaly scores come from the detector's deque
    anomaly_times = times[-len(anomaly_detector.anomaly_scores):] # Match length
    vis_objects['lines']['anomaly_score'].set_data(anomaly_times, list(anomaly_detector.anomaly_scores))
    vis_objects['lines']['phone'].set_data(times, list(visualization_data['phone_detected']))

    # Update landmark lines
    for i in range(settings.NUM_LANDMARKS):
        if f'lm{i}_x' in vis_objects['lines'] and f'lm{i}_x' in visualization_data:
            vis_objects['lines'][f'lm{i}_x'].set_data(times, list(visualization_data[f'lm{i}_x']))
        if f'lm{i}_y' in vis_objects['lines'] and f'lm{i}_y' in visualization_data:
            vis_objects['lines'][f'lm{i}_y'].set_data(times, list(visualization_data[f'lm{i}_y']))

    # --- Update Gaze Scatter Plot ---
    ax_scatter = vis_objects['axes']['gaze_scatter']
    # Get valid (non-NaN) gaze points for scatter plot
    valid_gaze_indices = [i for i, (y, p) in enumerate(zip(visualization_data['yaw'], visualization_data['pitch'])) if y is not None and p is not None and not np.isnan(y) and not np.isnan(p)]
    valid_yaw = [visualization_data['yaw'][i] for i in valid_gaze_indices]
    valid_pitch = [visualization_data['pitch'][i] for i in valid_gaze_indices]

    if valid_yaw:
        vis_objects['scatter'].set_offsets(np.column_stack((valid_yaw, valid_pitch)))
        # Auto-scale scatter plot based on data and thresholds
        max_data_x = max(np.max(np.abs(valid_yaw)), gaze_duration_tracker.yaw_threshold) if valid_yaw else gaze_duration_tracker.yaw_threshold
        max_data_y = max(np.max(np.abs(valid_pitch)), gaze_duration_tracker.pitch_threshold) if valid_pitch else gaze_duration_tracker.pitch_threshold
        ax_scatter.set_xlim(-max_data_x * 1.2, max_data_x * 1.2)
        ax_scatter.set_ylim(-max_data_y * 1.2, max_data_y * 1.2)
    else:
         vis_objects['scatter'].set_offsets(np.empty((0, 2))) # Clear scatter if no valid data
         # Keep reasonable default limits if no data
         ax_scatter.set_xlim(-settings.DEFAULT_YAW_THRESHOLD*1.5, settings.DEFAULT_YAW_THRESHOLD*1.5)
         ax_scatter.set_ylim(-settings.DEFAULT_PITCH_THRESHOLD*1.5, settings.DEFAULT_PITCH_THRESHOLD*1.5)


    # Update threshold lines on scatter plot
    yaw_thresh = gaze_duration_tracker.yaw_threshold
    pitch_thresh = gaze_duration_tracker.pitch_threshold
    xlim = ax_scatter.get_xlim()
    ylim = ax_scatter.get_ylim()
    vis_objects['lines']['yaw_thresh_pos'].set_data([yaw_thresh, yaw_thresh], ylim)
    vis_objects['lines']['yaw_thresh_neg'].set_data([-yaw_thresh, -yaw_thresh], ylim)
    vis_objects['lines']['pitch_thresh_pos'].set_data(xlim, [pitch_thresh, pitch_thresh])
    vis_objects['lines']['pitch_thresh_neg'].set_data(xlim, [-pitch_thresh, -pitch_thresh])
    # Explicitly set data for threshold lines to ensure they span the updated axes
    vis_objects['lines']['yaw_thresh_pos'].set_ydata(ylim)
    vis_objects['lines']['yaw_thresh_neg'].set_ydata(ylim)
    vis_objects['lines']['pitch_thresh_pos'].set_xdata(xlim)
    vis_objects['lines']['pitch_thresh_neg'].set_xdata(xlim)


    # --- Update Off-Screen Bar Chart ---
    ax_offscreen = vis_objects['axes']['offscreen']
    # Get durations from tracker that meet the threshold
    recent_durations = list(gaze_duration_tracker.off_screen_durations)
    valid_durations = [d for d in recent_durations if d >= settings.OFF_SCREEN_DURATION_THRESHOLD]
    positions = np.arange(len(valid_durations))

    ax_offscreen.cla() # Clear previous bars
    ax_offscreen.set_title(f"Recent Off-Screen Events (>{settings.OFF_SCREEN_DURATION_THRESHOLD:.1f}s)")
    ax_offscreen.set_ylabel("Duration (s)")
    ax_offscreen.set_xlabel("Event Index")
    ax_offscreen.grid(True, axis='y', linestyle=':')
    if len(valid_durations) > 0:
        vis_objects['bar_container'] = ax_offscreen.bar(positions, valid_durations, color="teal")
        ax_offscreen.set_xticks(positions) # Label each bar
        # Ensure x-axis covers at least the number of bars configured, or the actual number if more
        ax_offscreen.set_xlim(left=-0.5, right=max(len(valid_durations) - 0.5, settings.OFF_SCREEN_PLOT_BAR_COUNT - 0.5))
        max_duration_val = max(valid_durations) if valid_durations else 5.0 # Default max height
        ax_offscreen.set_ylim(bottom=0, top=max_duration_val * 1.15) # Add some headroom
    else:
        # Set defaults if no valid durations to show
        ax_offscreen.set_xticks([])
        ax_offscreen.set_xlim(left=-0.5, right=settings.OFF_SCREEN_PLOT_BAR_COUNT - 0.5)
        ax_offscreen.set_ylim(bottom=0, top=5.0) # Default y-limit


    # --- Rescale Axes and Draw ---
    min_time = times[0] if times else 0
    max_time = times[-1] if times else 1

    # Update axes limits for all time-based plots
    time_axes_keys = ['gaze_ts', 'anomaly', 'phone', 'lm_x', 'lm_y']
    for key in time_axes_keys:
        ax = vis_objects['axes'][key]
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True) # Autoscale Y only
        ax.set_xlim(min_time, max_time) # Set X based on current time window

    # Redraw the figure
    try:
        vis_objects['fig'].canvas.draw_idle()
        vis_objects['fig'].canvas.flush_events()
    except Exception as e:
        print(f"Plotting Error: {e}") # Avoid crashing the main loop

def close_plots():
    """Closes the Matplotlib figure."""
    import os
    global vis_objects

    path = settings.LOG_FOLDER_EVENTS / "plots"
    path.makedir(exist_ok = True)
    plt.savefig( path)
    if settings.ENABLE_INTERACTIVE_PLOTS and vis_objects and vis_objects.get('fig'):
        plt.close(vis_objects['fig'])
        vis_objects = None
        print("Closed interactive plot window.")