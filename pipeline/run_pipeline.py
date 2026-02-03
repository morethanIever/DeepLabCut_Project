# pipeline/run_pipeline.py
import os
import uuid
import shutil
from pathlib import Path
import pandas as pd
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

from pipeline.render_overlay import render_annotated_video

#from pipeline.ML.ml_features import extract_ml_features
#from pipeline.ML.umap_plot import save_umap_plot

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
    roi=None,
    resize_to=None
):
    """
    Full analysis pipeline with caching.

    force_pose=True:
        - Re-run DeepLabCut pose estimation (VERY SLOW)
    force_analysis=True:
        - Recompute kinematics & behavior (FAST)
    """
    ensure_dirs()

    dlc_video_dir = r"C:\Users\leelab\Desktop\TestBehaviour-Eunhye-2025-12-29\videos"
    os.makedirs(dlc_video_dir, exist_ok=True)


    # ----------------------------
    # 0) Save uploaded video
    # ----------------------------
    if isinstance(uploaded_file, str):
        # Video is a path string (likely from temp/ after cropping/enhancing)
        source_path = uploaded_file
        video_filename = os.path.basename(source_path)
        stable_input_path = os.path.join(dlc_video_dir, video_filename)
        
        # COPY from temp to permanent DLC folder if it's not already there
        if os.path.abspath(source_path) != os.path.abspath(stable_input_path):
            logs.append(f"[WORKFLOW] Moving preprocessed video to stable path: {video_filename}")
            shutil.copy(source_path, stable_input_path)
        
        input_video = stable_input_path
    else:
        # Fresh upload object from Streamlit
        input_video = save_uploaded_video(uploaded_file)
        logs.append(f"[WORKFLOW] Saved fresh upload to: {input_video}")

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
    scale = None
    if roi is not None and resize_to is not None:
        crop_w, crop_h = roi[2], roi[3]
        new_w, new_h = resize_to
        scale = (new_w / crop_w, new_h / crop_h)
    # nop analysis
    nop_summary_csv = run_nop_analysis(
        kin_csv,
        beh_csv,
        out_dir="outputs/nop",
        roi=roi,
        scale=scale
    )
    logs.append(f"[NOP] Summary CSV saved: {nop_summary_csv}")
    

    nop_plot_path = plot_nop(
        kin_csv=kin_csv,
        nop_summary_csv=nop_summary_csv,
        object_left=(480, 130),
        object_right=(957, 130),
    )
    logs.append(f"[NOP] Validation plot saved: {nop_plot_path}")
    """
    # ----------------------------
    # 4) Render annotated video
    # ----------------------------
    ml_feat_csv = extract_ml_features(
        kin_csv,
        input_video,
        force=force_analysis
    )
    df_ml = pd.read_csv(ml_feat_csv)
    
    if not df_ml.empty:
        umap_path = save_umap_plot(
            df_ml, 
            input_video, 
            output_dir="outputs/plots/ml",
            color_col=None 
        )
    
    logs.append(f"[PLOT] Unsupervised UMAP map saved: {umap_path}")
    
    umap_plot_path, cluster_labels = save_umap_plot(df_ml, input_video)

    df_ml['visual_cluster'] = cluster_labels
    df_ml.to_csv(ml_feat_csv, index=False)

    logs.append(f"[OK] Behavioral clusters saved to: {ml_feat_csv}")"""
    out_video = cached_outvideo_path(input_video)

    ALL_VIDEO_MAPS = {
        'fad92c0398211bcb9c0ef182e0fe65cb': {0: 'Fast move', 1: 'Slow Sniffing', 2: 'Move', 3: 'Rest', 4: 'Grooming', 5: 'Active Sniffing', 6: 'Turning', 7: 'Rearing'},
        'bf57b37b35cd86006f143dbf36d41402': {0: 'Move', 1: 'Grooming', 2: 'Turning', 3: 'Rearing', 4: 'Fast Move', 5: 'Sniffing', 6: 'Fast Move', 7: 'Rest'},
        'e8c9ec435e1183ddfd78181e05ecfac1': {0: 'Rest', 1: 'Turning', 2: 'Sniffing', 3: 'Rearing', 4: 'Grooming', 5: 'Fast Move', 6: 'Move', 7: 'Fast Move'},
        '0abdd7ba277898156608d9b842e97d68': {0: 'Sniffing', 1: 'Grooming', 2: 'Fast move', 3: 'Turning', 4: 'Rest', 5: 'Rearing', 6: 'Move', 7: 'Grooming'}
    }

    # 5) Determine which map to use based on input video name
    video_id = Path(input_video).stem # Extracts the filename without extension
    video_id = video_id.split('_')[0]
    current_map = ALL_VIDEO_MAPS.get(video_id, None)

    if current_map:
        logs.append(f"[RENDER] Found verified map for {video_id}")
    else:
        logs.append(f"[WARNING] No verified map found for {video_id}. Showing raw Cluster IDs.")
    
    out_video = render_annotated_video(
        input_video=input_video,
        pose_csv=pose_csv,
        kin_csv=kin_csv,
        beh_csv=beh_csv,
        #ml_feat_csv=ml_feat_csv,
        logs=logs,
        out_path=out_video, # Uses the path determined by cache_utils
        label_map=current_map,
        roi=None
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
        "kin_csv": kin_csv,
        "out_video": out_video,
        "logs": logs,
        "speed_plot": speed_plot,
        "trajectory_plot": trajectory_plot,
        "trajectory_behavior": trajectory_behavior,
        "turning_rate_plot_path": turning_rate_plot_path,
        "trajectory_turning_plot": trajectory_turning_plot,
        "nop_plot": nop_plot,
        #"ml_features": ml_feat_csv,
        #"umap_plot": umap_path
    }
