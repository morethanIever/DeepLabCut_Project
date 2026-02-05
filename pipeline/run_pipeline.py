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
from pipeline.visualisation.trajectory_behavior import plot_trajectory_by_behavior
from pipeline.kinematics.turning_rate import compute_turning_rate
from pipeline.visualisation.turning_rate_plot import plot_turning_rate
from pipeline.visualisation.trajectory_turning_plot import plot_trajectory_with_turning_rate

from pipeline.nop.nop_analysis import run_nop_analysis
from pipeline.nop.nop_plot import plot_nop

from pipeline.render_overlay import render_annotated_video

from pipeline.ML.ml_features import extract_ml_features
#from pipeline.ML.umap_plot import save_umap_plot

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _copy_to_dir(src_path: str, dst_dir: str, *, dst_name: str | None = None) -> str:
    _ensure_dir(dst_dir)
    name = dst_name or os.path.basename(src_path)
    dst_path = os.path.join(dst_dir, name)
    if os.path.abspath(src_path) != os.path.abspath(dst_path):
        shutil.copy(src_path, dst_path)
    return dst_path





def run_full_pipeline(
    uploaded_file_or_path,
    logs: list,
    *,
    dlc_config_path: str,
    out_dir: str,
    force_pose: bool = False,
    force_analysis: bool = False,
    roi=None,
    resize_to=None,
    output_name: str | None = None,
):
    """
    Full analysis pipeline with caching.

    force_pose=True:
        - Re-run DeepLabCut pose estimation (VERY SLOW)
    force_analysis=True:
        - Recompute kinematics & behavior (FAST)
    """
    ensure_dirs(out_dir)

    if not dlc_config_path or not os.path.exists(dlc_config_path):
        raise RuntimeError(f"DLC config.yaml not found: {dlc_config_path}")

    out_dir = os.path.abspath(out_dir)
    vid_dir   = _ensure_dir(os.path.join(out_dir, "videos"))
    plots_dir = _ensure_dir(os.path.join(out_dir, "plots"))
    nop_dir   = _ensure_dir(os.path.join(out_dir, "nop"))
    ml_dir    = _ensure_dir(os.path.join(out_dir, "ml"))
    rend_dir  = _ensure_dir(os.path.join(out_dir, "rendered"))

    # ----------------------------
    # 0) Save uploaded video
    # ----------------------------
    if isinstance(uploaded_file_or_path, str):
        input_video = _copy_to_dir(uploaded_file_or_path, vid_dir, dst_name=output_name)
        logs.append(f"[WORKFLOW] Using video: {input_video}")
    else:
        # Streamlit UploadedFile
        _ensure_dir(vid_dir)
        input_video = os.path.join(vid_dir, output_name or uploaded_file_or_path.name)
        with open(input_video, "wb") as f:
            f.write(uploaded_file_or_path.getbuffer())
        logs.append(f"[WORKFLOW] Saved upload to: {input_video}")

    # ----------------------------
    # Cache key (prefer stable name)
    # ----------------------------
    cache_key = Path(output_name or input_video).stem

    # ----------------------------
    # 1) Pose (DeepLabCut) [SLOW]
    # ----------------------------
    pose_csv = run_deeplabcut_pose(
        input_video,
        logs,
        CONFIG_PATH=dlc_config_path,
        force=force_pose,
        out_dir=out_dir,
        cache_key=cache_key
    )
    logs.append(f"[OK] Pose CSV: {pose_csv}")

    # ----------------------------
    # 2) Kinematics [FAST]
    # ----------------------------
    kin_csv = compute_kinematics(
        pose_csv,
        input_video,
        logs,
        force=force_analysis,
        out_dir=out_dir,
        cache_key=cache_key
    )
    logs.append(f"[OK] Kinematics CSV: {kin_csv}")

    # ----------------------------
    # 3) Behavior [FAST]
    # ----------------------------
    beh_csv = classify_behavior(
        kin_csv,
        input_video,
        logs,
        force=force_analysis,
        out_dir=out_dir,
        cache_key=cache_key
    )
    logs.append(f"[OK] Behavior CSV: {beh_csv}")


    # turning rate
    turn_csv = compute_turning_rate(
        kin_csv,
        input_video,
        logs,
        force=force_analysis,
        out_dir=out_dir,
        cache_key=cache_key
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
        out_dir=nop_dir,
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
    
    # ----------------------------
    # 4) Render annotated video
    # ----------------------------
    ml_feat_csv = extract_ml_features(
        kin_csv,
        input_video,
        force=force_analysis,
        out_dir=out_dir,
        cache_key=cache_key
    )
    """df_ml = pd.read_csv(ml_feat_csv)
    
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
    out_video = os.path.join(rend_dir, f"{Path(input_video).stem}_annotated.mp4")
    
    
    """ALL_VIDEO_MAPS = {
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
    """
    out_video = render_annotated_video(
        input_video=input_video,
        pose_csv=pose_csv,
        kin_csv=kin_csv,
        beh_csv=beh_csv,
        ml_feat_csv=ml_feat_csv,
        logs=logs,
        out_path=out_video, # Uses the path determined by cache_utils
        #label_map=current_map,
        roi=None
    )


    speed_plot = plot_speed(kin_csv, out_dir=plots_dir) if "out_dir" in plot_speed.__code__.co_varnames else plot_speed(kin_csv)
    trajectory_plot = plot_trajectory(kin_csv, out_dir=plots_dir) if "out_dir" in plot_trajectory.__code__.co_varnames else plot_trajectory(kin_csv)
    trajectory_behavior = plot_trajectory_by_behavior(kin_csv, beh_csv, out_dir=plots_dir)  # 필요시 out_dir 적용
    turning_rate_plot_path = plot_turning_rate(turn_csv, out_dir=plots_dir)                # 필요시 out_dir 적용
    trajectory_turning_plot = plot_trajectory_with_turning_rate(kin_csv, turn_csv, out_dir=plots_dir, video_path=input_video)
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
        "ml_features": ml_feat_csv,
        #"umap_plot": umap_path
    }
