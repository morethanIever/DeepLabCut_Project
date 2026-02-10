import os
import shutil
from pathlib import Path
import glob
import cv2
from typing import Optional, List, Tuple
import configparser

from pipeline.simba_overlay import render_simba_overlay


def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _latest_file(patterns: List[str]) -> Optional[str]:
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat, recursive=True))
    valid = [p for p in candidates if os.path.exists(p)]
    return max(valid, key=os.path.getmtime) if valid else None


def _get_video_props(video_path: str) -> Tuple[float, Tuple[int, int]]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, (w, h)


def _stage_pose_csv(pose_csv: str, staging_dir: str, *, stem: str) -> str:
    _ensure_dir(staging_dir)
    dst = os.path.join(staging_dir, f"{stem}.csv")
    if os.path.abspath(pose_csv) != os.path.abspath(dst):
        shutil.copy(pose_csv, dst)
    return dst


def _ensure_project_bp_names(project_dir: str, pose_csv: str) -> Optional[str]:
    """
    Create SimBA project_bp_names.csv (one bodypart per line) if missing.
    """
    logs_dir = os.path.join(project_dir, "logs", "measures", "pose_configs", "bp_names")
    os.makedirs(logs_dir, exist_ok=True)
    out_path = os.path.join(logs_dir, "project_bp_names.csv")
    if os.path.exists(out_path):
        return out_path
    try:
        import pandas as pd
        df = pd.read_csv(pose_csv, header=[0, 1, 2])
        bodyparts = sorted(set(df.columns.get_level_values(1)))
        bodyparts = [
            bp for bp in bodyparts
            if isinstance(bp, str) and bp.lower() not in {"bodyparts", "coords", "scorer"}
        ]
        if not bodyparts:
            return None
        with open(out_path, "w", encoding="utf-8") as f:
            for bp in bodyparts:
                f.write(f"{bp}\n")
        return out_path
    except Exception:
        return None


def _copy_video(input_video: str, project_dir: str) -> str:
    videos_dir = os.path.join(project_dir, "videos")
    _ensure_dir(videos_dir)
    dst = os.path.join(videos_dir, os.path.basename(input_video))
    if os.path.abspath(input_video) != os.path.abspath(dst):
        shutil.copy(input_video, dst)
    return dst


def _resolve_project_dir(config_path: str) -> str:
    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")
    for section in ("General settings", "Project"):
        if cfg.has_section(section) and cfg.has_option(section, "project_path"):
            p = cfg.get(section, "project_path", fallback="").strip()
            if p:
                return os.path.abspath(p)
    return str(Path(config_path).resolve().parent)


def _normalize_pose_estimation_body_parts(config_path: str) -> None:
    """
    SimBA expects an integer count for pose_estimation_body_parts in this version.
    If it looks like a comma-separated list, replace with the count.
    """
    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")
    section = "create ensemble settings"
    key = "pose_estimation_body_parts"
    if not (cfg.has_section(section) and cfg.has_option(section, key)):
        return
    raw = cfg.get(section, key, fallback="").strip()
    allowed = {"4", "7", "8", "9", "14", "16", "user_defined", "AMBER"}
    if "," in raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if parts:
            count = str(len(parts))
            cfg[section][key] = count if count in allowed else "user_defined"
            with open(config_path, "w", encoding="utf-8") as f:
                cfg.write(f)
            return
    if raw and raw not in allowed:
        cfg[section][key] = "user_defined"
        with open(config_path, "w", encoding="utf-8") as f:
            cfg.write(f)


def _ensure_video_info_csv(project_dir: str, input_video: str, config_path: str) -> Optional[str]:
    """
    Create or update logs/video_info.csv for the current video.
    Uses video props and config hints.
    """
    logs_dir = os.path.join(project_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    out_path = os.path.join(logs_dir, "video_info.csv")

    fps, (w, h) = _get_video_props(input_video)
    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")
    px_per_mm = None
    distance_mm = None
    if cfg.has_section("Videos"):
        px_per_mm = cfg.get("Videos", "px_per_mm", fallback="").strip()
    if cfg.has_section("Frame settings"):
        distance_mm = cfg.get("Frame settings", "distance_mm", fallback="").strip()

    try:
        px_per_mm_val = float(px_per_mm) if px_per_mm not in (None, "", "None") else 0.0
    except Exception:
        px_per_mm_val = 0.0
    try:
        distance_mm_val = float(distance_mm) if distance_mm not in (None, "", "None") else 1.0
    except Exception:
        distance_mm_val = 1.0

    import pandas as pd
    video_name = os.path.splitext(os.path.basename(input_video))[0]
    row = {
        "Video": video_name,
        "fps": float(fps),
        "Resolution_width": float(w),
        "Resolution_height": float(h),
        "Distance_in_mm": float(distance_mm_val),
        "pixels/mm": float(px_per_mm_val),
    }

    if os.path.exists(out_path):
        df = pd.read_csv(out_path)
        if "Video" in df.columns and (df["Video"].astype(str) == row["Video"]).any():
            return out_path
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(out_path, index=False)
    return out_path


def _find_latest_machine_results(project_dir: str, *, stem: str) -> Optional[str]:
    root = Path(project_dir) / "csv"
    preferred = [
        root / "machine_results" / f"*{stem}*.csv",
        root / "classifier_validation" / f"*{stem}*.csv",
        root / "inference" / f"*{stem}*.csv",
        root / "**" / f"*{stem}*.csv",
    ]
    return _latest_file([str(p) for p in preferred])


def run_simba_pipeline(
    *,
    config_path: str,
    pose_csv: str,
    input_video: str,
    logs: list,
    out_dir: str,
    force: bool = False,
    interpolation_setting: Optional[str] = None,
    smoothing_setting: Optional[str] = None,
    smoothing_time_ms: Optional[int] = None,
    px_per_mm: Optional[float] = None,
    render_video: bool = True,
) -> dict:
    """
    Run SimBA full pipeline using the SimBA Python API.
    Requires SimBA to be installed in the environment.
    """
    if not config_path or not os.path.exists(config_path):
        raise RuntimeError(f"[SimBA] config not found: {config_path}")

    _normalize_pose_estimation_body_parts(config_path)

    try:
        from simba.outlier_tools.skip_outlier_correction import OutlierCorrectionSkipper
        from simba.utils.cli.cli_tools import feature_extraction_runner
        from simba.model.inference_batch import InferenceBatch
        try:
            from simba.video_processors.batch_process_video import set_video_parameters
        except Exception:
            set_video_parameters = None

        import_func = None
        import_mode = None
        try:
            from simba.pose_importers.dlc_importer_csv import import_multiple_dlc_tracking_csv_file
            import_func = import_multiple_dlc_tracking_csv_file
            import_mode = "multi"
        except Exception:
            try:
                from simba.pose_importers.dlc_importer_csv import import_dlc_csv, import_dlc_csv_data
                import_func = import_dlc_csv
                import_mode = "single"
            except Exception:
                import_func = None
                import_mode = None
    except Exception as e:
        raise RuntimeError(
            "SimBA import failed in the active environment. "
            f"Import error: {e}"
        ) from e

    project_dir = _resolve_project_dir(config_path)
    stem = Path(input_video).stem

    _copy_video(input_video, project_dir)
    _ensure_project_bp_names(project_dir, pose_csv)
    _ensure_video_info_csv(project_dir, input_video, config_path)
    # Ensure SimBA output dirs exist
    os.makedirs(os.path.join(project_dir, "csv", "machine_results"), exist_ok=True)

    staging_root = _ensure_dir(os.path.join(out_dir, "simba", "staging"))
    staging_dir = _ensure_dir(os.path.join(staging_root, stem))
    # ensure staging dir is clean to avoid importing unrelated CSVs
    for p in glob.glob(os.path.join(staging_dir, "*.csv")):
        try:
            os.remove(p)
        except Exception:
            pass
    staged_pose = _stage_pose_csv(pose_csv, staging_dir, stem=stem)

    if force:
        input_csv_dir = os.path.join(project_dir, "csv", "input_csv")
        for p in glob.glob(os.path.join(input_csv_dir, f"*{stem}*.csv")):
            try:
                os.remove(p)
            except Exception:
                pass

    interp = interpolation_setting or "None"
    smooth = smoothing_setting or "None"
    smooth_time = smoothing_time_ms or 0

    logs.append(f"[SimBA] Import DLC CSV with interpolation={interp}, smoothing={smooth}")
    if import_func is None:
        raise RuntimeError("No compatible SimBA DLC import function found in this version.")
    input_csv_dir = os.path.join(project_dir, "csv", "input_csv")
    expected_csv = os.path.join(input_csv_dir, f"{stem}.csv")
    if (not force) and os.path.exists(expected_csv):
        logs.append(f"[SimBA] Input CSV exists. Skipping import: {expected_csv}")
    else:
        if import_mode == "multi":
            import_func(
                config_path=config_path,
                data_dir=os.path.dirname(staged_pose),
                interpolation_setting=interp,
                smoothing_setting=smooth,
                smoothing_time=smooth_time,
            )
        else:
            # Older SimBA: import folder or file without interpolation/smoothing options
            import_func(
                config_path=config_path,
                source=os.path.dirname(staged_pose),
            )

    logs.append("[SimBA] Skipping outlier correction")
    OutlierCorrectionSkipper(config_path=config_path).run()

    fps, (w, h) = _get_video_props(input_video)
    if px_per_mm and px_per_mm > 0:
        if set_video_parameters is None:
            logs.append("[SimBA] set_video_parameters not available in this SimBA version. Skipping.")
        else:
            logs.append(f"[SimBA] Set video params: px_per_mm={px_per_mm}, fps={fps:.2f}, res=({w},{h})")
            set_video_parameters(
                config_path=config_path,
                px_per_mm=px_per_mm,
                fps=fps,
                resolution=f"{w}x{h}",
            )
    else:
        logs.append("[SimBA] Skipping set_video_parameters (px_per_mm not set)")

    logs.append("[SimBA] Feature extraction")
    feature_extraction_runner(config_path=config_path)

    logs.append("[SimBA] Inference (preconfigured models)")
    InferenceBatch(config_path=config_path).run()

    machine_csv = _find_latest_machine_results(project_dir, stem=stem)
    if machine_csv:
        logs.append(f"[SimBA] Machine results: {machine_csv}")
    else:
        logs.append("[SimBA] Machine results not found after inference.")

    simba_video = None
    if render_video and machine_csv:
        out_video = os.path.join(out_dir, "simba", f"{stem}_simba_overlay.mp4")
        simba_video = render_simba_overlay(
            input_video=input_video,
            simba_csv=machine_csv,
            out_path=out_video,
        )
        if simba_video:
            logs.append(f"[SimBA] Overlay video saved: {simba_video}")

    return {
        "simba_machine_csv": machine_csv,
        "simba_overlay_video": simba_video,
    }
