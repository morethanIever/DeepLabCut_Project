# pipeline/run_pipeline.py
import os
import uuid
from pathlib import Path

from pipeline.pose_dlc import run_deeplabcut_pose
from pipeline.kinematics.kinematics import compute_kinematics
from pipeline.behavior import classify_behavior
from pipeline.render_overlay import render_annotated_video
from pipeline.cache_utils import (
    ensure_dirs,
    cached_outvideo_path,
)

from pipeline.visualisation.speed_plot import plot_speed
from pipeline.visualisation.trajectory_plot import plot_trajectory
from pipeline.trajectory_behavior import plot_trajectory_by_behavior
from pipeline.kinematics.turning_rate import compute_turning_rate
from pipeline.visualisation.turning_rate_plot import plot_turning_rate
from pipeline.visualisation.trajectory_turning_plot import plot_trajectory_with_turning_rate

from pipeline.nop.nop_analysis import run_nop_analysis
from pipeline.nop.nop_plot import plot_nop

#from pipeline.nop.nop_time_binned_short import run_nop_time_binned_short
#from pipeline.nop.nop_time_binned_short_plot import plot_nop_time_binned_short

from pipeline.ML.ml_features import extract_ml_features

def save_uploaded_video(uploaded_file) -> str:
    import shutil

    dlc_video_dir = r"C:\Users\leelab\Desktop\TestBehaviour-Eunhye-2025-12-29\videos"
    os.makedirs(dlc_video_dir, exist_ok=True)

    video_name = uploaded_file.name
    dlc_video_path = os.path.join(dlc_video_dir, video_name)

    with open(dlc_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return dlc_video_path



def run_full_pipeline(
    uploaded_file,
    logs: list,
    *,
    force_pose: bool = False,
    force_analysis: bool = False,
):
    """
    Full analysis pipeline with caching.

    force_pose=True:
        - Re-run DeepLabCut pose estimation (VERY SLOW)
    force_analysis=True:
        - Recompute kinematics & behavior (FAST)
    """
    ensure_dirs()

    # ----------------------------
    # 0) Save uploaded video
    # ----------------------------
    input_video = save_uploaded_video(uploaded_file)
    logs.append(f"[OK] Saved input video: {input_video}")

    # ----------------------------
    # 1) Pose (DeepLabCut) [SLOW]
    # ----------------------------
    pose_csv = run_deeplabcut_pose(
        input_video,
        logs,
        force=force_pose
    )
    logs.append(f"[OK] Pose CSV: {pose_csv}")

    # ----------------------------
    # 2) Kinematics [FAST]
    # ----------------------------
    kin_csv = compute_kinematics(
        pose_csv,
        input_video,
        logs,
        force=force_analysis
    )
    logs.append(f"[OK] Kinematics CSV: {kin_csv}")

    # ----------------------------
    # 3) Behavior [FAST]
    # ----------------------------
    beh_csv = classify_behavior(
        kin_csv,
        input_video,
        logs,
        force=force_analysis
    )
    logs.append(f"[OK] Behavior CSV: {beh_csv}")


    # turning rate
    turn_csv = compute_turning_rate(
        kin_csv,
        input_video,
        logs,
        force=force_analysis
    )
    
    logs.append(f"[OK] Turning rate CSV: {turn_csv}")
    
    # nop analysis
    nop_summary_csv = run_nop_analysis(
        kin_csv,
        beh_csv,
        out_dir="outputs/nop"
    )
    logs.append(f"[NOP] Summary CSV saved: {nop_summary_csv}")
    
    """
    video_duration = len(pd.read_csv(kin_csv)) / 30
    if video_duration < 30:
        nop_binned_csv = run_nop_time_binned_short(
            kin_csv,
            out_dir = "output/nop"
        )
        nop_plot = plot_nop_time_binned_short(nop_binned_csv)
    else:
        nop_binned_csv = run_nop_time_binned_short(kin_csv)
    logs.append(f"[NOP] Time-binned CSV saved: {nop_binned_csv}")
    """
    
    nop_plot_path = plot_nop(
        kin_csv=kin_csv,
        nop_summary_csv=nop_summary_csv,
        object_left=(480, 130),
        object_right=(957, 130),
    )
    logs.append(f"[NOP] Validation plot saved: {nop_plot_path}")
    
    # ----------------------------
    # 4) Render annotated video
    # ----------------------------
    out_video = cached_outvideo_path(input_video)

    render_annotated_video(
        input_video=input_video,
        pose_csv=pose_csv,
        kin_csv=kin_csv,
        beh_csv=beh_csv,
        logs=logs,
        out_path=out_video,
    )
    
    ml_feat_csv = extract_ml_features(
        kin_csv,
        input_video,
        force=force_analysis
    )
    
    speed_plot = plot_speed(kin_csv)
    trajectory_plot = plot_trajectory(kin_csv)
    trajectory_behavior = plot_trajectory_by_behavior(kin_csv, beh_csv)
    turning_rate_plot_path = plot_turning_rate(turn_csv)
    trajectory_turning_plot = plot_trajectory_with_turning_rate(kin_csv, turn_csv)
    nop_plot = nop_plot_path
    
    logs.append(f"[PLOT] Speed plot saved: {speed_plot}")
    logs.append(f"[PLOT] Trajectory plot saved: {trajectory_plot}")
    logs.append(f"[PLOT] Trajectory by behaviour plot saved: {trajectory_behavior}")
    logs.append(f"[PLOT] Turning rate plot saved: {turning_rate_plot_path}")
    logs.append(f"[PLOT] Trajectory turning plot saved: {trajectory_turning_plot}")
    logs.append(f"[PLOT] NOP plot saved: {nop_plot}")
    logs.append(f"[OK] Output video: {out_video}")


    return {
        "input_video": input_video,
        "out_video": out_video,
        "logs": logs,
        "speed_plot": speed_plot,
        "trajectory_plot": trajectory_plot,
        "trajectory_behavior": trajectory_behavior,
        "turning_rate_plot_path": turning_rate_plot_path,
        "trajectory_turning_plot": trajectory_turning_plot,
        "nop_plot": nop_plot,
        "ml_features": ml_feat_csv,
    }
