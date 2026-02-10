import os
import streamlit as st
import glob


def init_session_state():
    defaults = {
        "uploaded_file_id": None, "input_video_path": None, "output_video": None,
        "logs": [], "page": "main", "kin_csv_path": None, "crop_roi": None, "resize_to": None,
        "clahe_clip": None,
        "speed_plot": None, "trajectory_plot": None, "trajectory_behavior": None,
        "turning_rate_plot_path": None, "trajectory_turning_plot": None, "nop_plot": None,
        "simba_machine_csv": None, "simba_overlay_video": None,
        "roi_list": [], "roi_radius": 80, "roi_result_df": None, "roi_result_plot": None,
        "active_project": None, "active_profile": None,
        "original_video_name": None,
        "experiment_type": "Open Field", "camera_view": "Top View"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_analysis_state():
    keys = ["output_video", "logs", "speed_plot", "trajectory_plot", "trajectory_behavior",
            "turning_rate_plot_path", "trajectory_turning_plot", "nop_plot", "kin_csv_path",
            "simba_machine_csv", "simba_overlay_video",
            "roi_result_df", "roi_result_plot"]
    for k in keys: st.session_state[k] = None
    st.session_state.logs = []

def _get_latest_file(pattern_list):
    all_files = []
    for p in pattern_list:
        all_files.extend(glob.glob(str(p), recursive=True))
    valid_files = [f for f in all_files if os.path.exists(f)]
    return max(valid_files, key=os.path.getmtime) if valid_files else None
