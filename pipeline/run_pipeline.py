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
from projects.projects import load_profile
from pipeline.cache_utils import (
    ensure_dirs,
    cached_outvideo_path,
)

from pipeline.visualisation.speed_plot import plot_speed
# from pipeline.visualisation.speed_plotly import animate_speed_plotly
from pipeline.visualisation.trajectory_plot import plot_trajectory
from pipeline.visualisation.trajectory_speed_anim import animate_trajectory_mp4
from pipeline.visualisation.trajectory_behavior import plot_trajectory_by_behavior
from pipeline.kinematics.turning_rate import compute_turning_rate
from pipeline.visualisation.turning_rate_plot import plot_turning_rate
from pipeline.visualisation.trajectory_turning_plot import plot_trajectory_with_turning_rate

from pipeline.nop.nop_analysis import run_nop_analysis
from pipeline.nop.nop_plot import plot_nop

from pipeline.ML.ml_features import extract_ml_features
from pipeline.simba_backend import run_simba_pipeline
#from pipeline.ML.umap_plot import save_umap_plot

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


from typing import Optional


def _copy_to_dir(src_path: str, dst_dir: str, *, dst_name: Optional[str] = None) -> str:
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
    run_simba: bool = False,
    simba_config_path: Optional[str] = None,
    simba_options: Optional[dict] = None,
    behavior_mode: str = "auto",
    kinematics_csv: Optional[str] = None,
    project_name: Optional[str] = None,
    projects_root: str = "projects",
    roi=None,
    resize_to=None,
    output_name: Optional[str] = None,
    dlc_log_callback=None,
    render_pcutoff: float | None = None,
    steps: Optional[list[str]] = None,
    input_video_override: Optional[str] = None,
    pose_csv: Optional[str] = None,
    kin_csv: Optional[str] = None,
    beh_csv: Optional[str] = None,
    turn_csv: Optional[str] = None,
    ml_feat_csv: Optional[str] = None,
    simba_machine_csv: Optional[str] = None,
):
    """
    Full analysis pipeline with caching.

    force_pose=True:
        - Re-run DeepLabCut pose estimation (VERY SLOW)
    force_analysis=True:
        - Recompute kinematics & behavior (FAST)
    """
    import time
    def _log(msg: str) -> None:
        logs.append(msg)
        if dlc_log_callback:
            try:
                dlc_log_callback(msg)
            except Exception:
                pass

    ensure_dirs(out_dir)
    steps_set = set([s.lower() for s in steps]) if steps else None
    def _run_step(name: str) -> bool:
        return steps_set is None or name in steps_set

    if project_name:
        prof = load_profile(projects_root, project_name)
        if not dlc_config_path:
            dlc_config_path = prof.get("dlc", {}).get("config_path", "")
        if not simba_config_path:
            simba_config_path = prof.get("simba", {}).get("config_path", "")

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
    _log("[PIPELINE] Step 0: prepare input video")
    t0 = time.time()
    if _run_step("prepare"):
        if isinstance(uploaded_file_or_path, str):
            input_video = _copy_to_dir(uploaded_file_or_path, vid_dir, dst_name=output_name)
            _log(f"[WORKFLOW] Using video: {input_video}")
        else:
            # Streamlit UploadedFile
            _ensure_dir(vid_dir)
            input_video = os.path.join(vid_dir, output_name or uploaded_file_or_path.name)
            with open(input_video, "wb") as f:
                f.write(uploaded_file_or_path.getbuffer())
            _log(f"[WORKFLOW] Saved upload to: {input_video}")
        _log(f"[PIPELINE] Step 0 done in {time.time() - t0:.1f}s")
    else:
        if input_video_override:
            input_video = input_video_override
        elif isinstance(uploaded_file_or_path, str) and os.path.exists(uploaded_file_or_path):
            input_video = uploaded_file_or_path
        else:
            raise RuntimeError("Input video is required when 'prepare' step is skipped.")
        _log(f"[WORKFLOW] Using existing video: {input_video}")

    # ----------------------------
    # Cache key (prefer stable name)
    # ----------------------------
    cache_key = Path(output_name or input_video).stem

    # ----------------------------
    # 1) Pose (DeepLabCut) [SLOW]
    # ----------------------------
    _log("[PIPELINE] Step 1: DLC pose estimation")
    t1 = time.time()
    if _run_step("pose"):
        pose_csv = run_deeplabcut_pose(
            input_video,
            logs,
            CONFIG_PATH=dlc_config_path,
            force=force_pose,
            out_dir=out_dir,
            cache_key=cache_key,
            log_callback=dlc_log_callback,
        )
        _log(f"[OK] Pose CSV: {pose_csv}")
        _log(f"[PIPELINE] Step 1 done in {time.time() - t1:.1f}s")
    else:
        if not pose_csv or not os.path.exists(pose_csv):
            raise RuntimeError("Pose CSV is required when 'pose' step is skipped.")
        _log(f"[OK] Using existing Pose CSV: {pose_csv}")

    # ----------------------------
    # 2) Kinematics [FAST]
    # ----------------------------
    map_json = Path(out_dir).parent / "assets" / "kinematics_map.json"
    map_yaml = Path(out_dir).parent / "assets" / "kinematics_map.yaml"
    map_yml = Path(out_dir).parent / "assets" / "kinematics_map.yml"
    mapping_path = None
    if map_json.exists():
        mapping_path = str(map_json)
    elif map_yaml.exists():
        mapping_path = str(map_yaml)
    elif map_yml.exists():
        mapping_path = str(map_yml)

    _log("[PIPELINE] Step 2: Kinematics")
    t2 = time.time()
    if _run_step("kinematics"):
        kin_source_csv = kinematics_csv or pose_csv
        if kinematics_csv:
            _log(f"[KIN] Using provided kinematics source CSV: {kin_source_csv}")
        kin_csv = compute_kinematics(
            kin_source_csv,
            input_video,
            logs,
            force=force_analysis,
            out_dir=out_dir,
            cache_key=cache_key,
            mapping_path=mapping_path,
        )
        _log(f"[OK] Kinematics CSV: {kin_csv}")
        _log(f"[PIPELINE] Step 2 done in {time.time() - t2:.1f}s")
    else:
        if not kin_csv or not os.path.exists(kin_csv):
            raise RuntimeError("Kinematics CSV is required when 'kinematics' step is skipped.")
        _log(f"[OK] Using existing Kinematics CSV: {kin_csv}")

    # ----------------------------
    # 3) Behavior [FAST]
    # ----------------------------
    _log("[PIPELINE] Step 3: Behavior")
    t3 = time.time()
    if _run_step("behavior"):
        if behavior_mode == "auto":
            beh_csv = classify_behavior(
                kin_csv,
                input_video,
                logs,
                force=force_analysis,
                out_dir=out_dir,
                cache_key=cache_key
            )
            _log(f"[OK] Behavior CSV: {beh_csv}")
        elif behavior_mode in {"skip", "manual"}:
            beh_csv = None
            _log(f"[BEH] Skipped behavior (mode={behavior_mode}).")
        else:
            raise ValueError(f"Unknown behavior_mode: {behavior_mode}. Use 'auto', 'skip', or 'manual'.")
        _log(f"[PIPELINE] Step 3 done in {time.time() - t3:.1f}s")
    else:
        if beh_csv and os.path.exists(beh_csv):
            _log(f"[OK] Using existing Behavior CSV: {beh_csv}")
        else:
            _log("[BEH] Skipped (no existing behavior CSV provided).")


    # turning rate
    _log("[PIPELINE] Step 4: Turning rate")
    t4 = time.time()
    if _run_step("turning"):
        turn_csv = compute_turning_rate(
            kin_csv,
            input_video,
            logs,
            force=force_analysis,
            out_dir=out_dir,
            cache_key=cache_key
        )
        _log(f"[OK] Turning rate CSV: {turn_csv}")
        _log(f"[PIPELINE] Step 4 done in {time.time() - t4:.1f}s")
    else:
        if not turn_csv or not os.path.exists(turn_csv):
            raise RuntimeError("Turning-rate CSV is required when 'turning' step is skipped.")
        _log(f"[OK] Using existing Turning-rate CSV: {turn_csv}")

    scale = None
    if roi is not None and resize_to is not None:
        crop_w, crop_h = roi[2], roi[3]
        new_w, new_h = resize_to
        scale = (new_w / crop_w, new_h / crop_h)
        
    # nop analysis
    nop_summary_csv = None
    nop_plot_path = None
    _log("[PIPELINE] Step 5: NOP")
    t5 = time.time()
    if _run_step("nop"):
        if beh_csv:
            nop_summary_csv = run_nop_analysis(
                kin_csv,
                beh_csv,
                out_dir=nop_dir,
                roi=roi,
                scale=scale
            )
            _log(f"[NOP] Summary CSV saved: {nop_summary_csv}")

            nop_plot_path = plot_nop(
                kin_csv=kin_csv,
                nop_summary_csv=nop_summary_csv,
                object_left=(480, 130),
                object_right=(957, 130),
            )
            _log(f"[NOP] Validation plot saved: {nop_plot_path}")
        else:
            _log("[NOP] Skipped (behavior required).")
        _log(f"[PIPELINE] Step 5 done in {time.time() - t5:.1f}s")
    else:
        _log("[NOP] Skipped.")

    _log("[PIPELINE] Step 6: SimBA (optional)")
    t6 = time.time()
    simba_results = {"simba_machine_csv": None, "simba_overlay_video": None}
    if _run_step("simba"):
        if not run_simba:
            _log("[SimBA] Skipped: run_simba=False.")
        elif not simba_config_path:
            _log("[SimBA] Skipped: config path not set.")
        else:
            try:
                simba_results = run_simba_pipeline(
                    config_path=simba_config_path,
                    pose_csv=pose_csv,
                    input_video=input_video,
                    logs=logs,
                    out_dir=out_dir, 
                    kin_csv=kin_csv,
                    **(simba_options or {}),
                )
            except Exception as e:
                _log(f"[SimBA] Failed: {e}")
        _log(f"[PIPELINE] Step 6 done in {time.time() - t6:.1f}s")
    else:
        _log("[SimBA] Skipped.")

    # ----------------------------
    # 4) Render annotated video
    # ----------------------------
    _log("[PIPELINE] Step 7: Render + ML features")
    t7 = time.time()
    if _run_step("render"):
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
            roi=None,
            pcutoff=render_pcutoff,
        )
        _log(f"[PIPELINE] Step 7 done in {time.time() - t7:.1f}s")
    else:
        out_video = None
        _log("[RENDER] Skipped.")


    speed_plot = None
    speed_ani = None
    trajectory_plot = None
    trajectory_speed_ani = None
    trajectory_behavior = None
    turning_rate_plot_path = None
    trajectory_turning_plot = None
    nop_plot = nop_plot_path
    if _run_step("plots"):
        speed_plot = plot_speed(kin_csv, out_dir=plots_dir) if "out_dir" in plot_speed.__code__.co_varnames else plot_speed(kin_csv)
        speed_ani = None
        trajectory_plot = plot_trajectory(kin_csv, out_dir=plots_dir) if "out_dir" in plot_trajectory.__code__.co_varnames else plot_trajectory(kin_csv)
        trajectory_speed_ani = animate_trajectory_mp4(kin_csv, out_dir=plots_dir) if "out_dir" in animate_trajectory_mp4.__code__.co_varnames else animate_trajectory_mp4(kin_csv)
        simba_beh_source = None
        if isinstance(simba_results, dict):
            simba_beh_source = simba_results.get("simba_machine_csv")
        if not simba_beh_source:
            simba_beh_source = simba_machine_csv
        if simba_beh_source and os.path.exists(simba_beh_source):
            trajectory_behavior = plot_trajectory_by_behavior(kin_csv, simba_beh_source, out_dir=plots_dir)
        elif beh_csv:
            trajectory_behavior = plot_trajectory_by_behavior(kin_csv, beh_csv, out_dir=plots_dir)
        turning_rate_plot_path = plot_turning_rate(turn_csv, out_dir=plots_dir) if turn_csv else None
        trajectory_turning_plot = plot_trajectory_with_turning_rate(kin_csv, turn_csv, out_dir=plots_dir, video_path=input_video) if turn_csv else None

        _log(f"[PLOT] Speed plot saved: {speed_plot}")
        # Plotly speed animation disabled to avoid background generation overhead.
        _log(f"[PLOT] Trajectory plot saved: {trajectory_plot}")
        _log(f"[PLOT] Trajectory speed plot saved: {trajectory_speed_ani}")
        if trajectory_behavior:
            _log(f"[PLOT] Trajectory by behaviour plot saved: {trajectory_behavior}")
        if turning_rate_plot_path:
            _log(f"[PLOT] Turning rate plot saved: {turning_rate_plot_path}")
        if trajectory_turning_plot:
            _log(f"[PLOT] Trajectory turning plot saved: {trajectory_turning_plot}")
        if nop_plot:
            _log(f"[PLOT] NOP plot saved: {nop_plot}")
        if out_video:
            _log(f"[OK] Output video: {out_video}")
    else:
        _log("[PLOT] Skipped.")


    return {
        "input_video": input_video,
        "pose_csv": pose_csv,
        "kin_csv": kin_csv,
        "beh_csv": beh_csv,
        "turn_csv": turn_csv,
        "out_video": out_video,
        "logs": logs,
        "speed_plot": speed_plot,
        "speed anim": speed_ani,
        "trajectory_plot": trajectory_plot,
        "trajectory_speed_ani": trajectory_speed_ani,
        "trajectory_behavior": trajectory_behavior,
        "turning_rate_plot_path": turning_rate_plot_path,
        "trajectory_turning_plot": trajectory_turning_plot,
        "nop_plot": nop_plot,
        "ml_features": ml_feat_csv,
        "simba_machine_csv": simba_results.get("simba_machine_csv"),
        "simba_overlay_video": simba_results.get("simba_overlay_video"),
        #"umap_plot": umap_path
    }
