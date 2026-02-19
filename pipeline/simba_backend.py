import os
import shutil
import json
import subprocess
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


def _get_conda_exe() -> str:
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        return conda_exe
    candidate = os.path.join(os.path.expanduser("~"), "anaconda3", "Scripts", "conda.exe")
    if os.path.exists(candidate):
        return candidate
    return "conda"


def _run_simba_pipeline_subprocess(
    *,
    config_path: str,
    pose_csv: str,
    input_video: str,
    out_dir: str,
    force: bool,
    kin_csv: Optional[str],
    augment_with_kinematics: bool,
    interpolation_setting: Optional[str],
    smoothing_setting: Optional[str],
    smoothing_time_ms: Optional[int],
    px_per_mm: Optional[float],
    render_video: bool,
) -> None:
    logs: list = []
    try:
        result = run_simba_pipeline(
            config_path=config_path,
            pose_csv=pose_csv,
            input_video=input_video,
            logs=logs,
            out_dir=out_dir,
            force=force,
            kin_csv=kin_csv,
            augment_with_kinematics=augment_with_kinematics,
            interpolation_setting=interpolation_setting,
            smoothing_setting=smoothing_setting,
            smoothing_time_ms=smoothing_time_ms,
            px_per_mm=px_per_mm,
            render_video=render_video,
            allow_external=False,
        )
        payload = {"ok": True, "result": result, "logs": logs}
    except Exception as e:
        payload = {"ok": False, "error": str(e), "logs": logs}
    print(json.dumps(payload))


def _run_simba_pipeline_external(
    *,
    config_path: str,
    pose_csv: str,
    input_video: str,
    out_dir: str,
    force: bool,
    kin_csv: Optional[str],
    augment_with_kinematics: bool,
    interpolation_setting: Optional[str],
    smoothing_setting: Optional[str],
    smoothing_time_ms: Optional[int],
    px_per_mm: Optional[float],
    render_video: bool,
    conda_env: str,
) -> dict:
    conda_exe = _get_conda_exe()
    env = os.environ.copy()
    env.update(
        {
            "SIMBA_CFG_PATH": str(config_path),
            "SIMBA_POSE_CSV": str(pose_csv),
            "SIMBA_INPUT_VIDEO": str(input_video),
            "SIMBA_OUT_DIR": str(out_dir),
            "SIMBA_FORCE": "1" if force else "0",
            "SIMBA_KIN_CSV": str(kin_csv or ""),
            "SIMBA_AUGMENT": "1" if augment_with_kinematics else "0",
            "SIMBA_INTERP": str(interpolation_setting or ""),
            "SIMBA_SMOOTH": str(smoothing_setting or ""),
            "SIMBA_SMOOTH_MS": str(smoothing_time_ms or ""),
            "SIMBA_PX_PER_MM": str(px_per_mm or ""),
            "SIMBA_RENDER": "1" if render_video else "0",
        }
    )
    code = (
        "import os; "
        "from pipeline.simba_backend import _run_simba_pipeline_subprocess as f; "
        "f(config_path=os.environ['SIMBA_CFG_PATH'], "
        "pose_csv=os.environ['SIMBA_POSE_CSV'], "
        "input_video=os.environ['SIMBA_INPUT_VIDEO'], "
        "out_dir=os.environ['SIMBA_OUT_DIR'], "
        "force=os.environ.get('SIMBA_FORCE')=='1', "
        "kin_csv=(os.environ.get('SIMBA_KIN_CSV') or None), "
        "augment_with_kinematics=(os.environ.get('SIMBA_AUGMENT')=='1'), "
        "interpolation_setting=(os.environ.get('SIMBA_INTERP') or None), "
        "smoothing_setting=(os.environ.get('SIMBA_SMOOTH') or None), "
        "smoothing_time_ms=(int(os.environ['SIMBA_SMOOTH_MS']) if os.environ.get('SIMBA_SMOOTH_MS') else None), "
        "px_per_mm=(float(os.environ['SIMBA_PX_PER_MM']) if os.environ.get('SIMBA_PX_PER_MM') else None), "
        "render_video=(os.environ.get('SIMBA_RENDER')!='0'))"
    )
    cmd = [conda_exe, "run", "-n", conda_env, "python", "-c", code]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    # Try to parse JSON from stdout
    payload = None
    if result.stdout:
        for line in reversed(result.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    payload = json.loads(line)
                    break
                except Exception:
                    continue
    if payload:
        if not payload.get("ok", False):
            msg = payload.get("error") or "SimBA subprocess failed"
            raise RuntimeError(msg)
        return payload.get("result", {})
    msg = (result.stderr or result.stdout or "").strip() or "SimBA subprocess failed"
    raise RuntimeError(msg)
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


def _ensure_video_info_csv(
    project_dir: str,
    input_video: str,
    config_path: str,
    *,
    video_name_override: Optional[str] = None,
) -> Optional[str]:
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
    video_name = video_name_override or os.path.splitext(os.path.basename(input_video))[0]
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


def _align_features_to_model(
    project_dir: str,
    stem: str,
    model_paths: List[str],
    *,
    logs: Optional[list] = None,
) -> None:
    """
    Align features_extracted columns to the model's expected feature list.
    Creates an in-place aligned CSV for inference if needed.
    """
    if not model_paths:
        if logs is not None:
            logs.append(f"[SimBA] Align skipped (no model paths) for {stem}")
        return
    try:
        import joblib
        import pandas as pd
    except Exception:
        if logs is not None:
            logs.append(f"[SimBA] Align skipped (joblib/pandas missing) for {stem}")
        return

    model_path = next((p for p in model_paths if p and os.path.exists(p)), None)
    if not model_path:
        if logs is not None:
            logs.append(f"[SimBA] Align skipped (model not found) for {stem}")
        return

    try:
        clf = joblib.load(model_path)
        feats = getattr(clf, "feature_names_in_", None)
        if feats is None:
            if logs is not None:
                logs.append(f"[SimBA] Align skipped (model has no feature_names_in_) for {stem}")
            return
        feats = list(feats)
    except Exception:
        if logs is not None:
            logs.append(f"[SimBA] Align skipped (model load failed) for {stem}")
        return

    feat_csv = os.path.join(project_dir, "csv", "features_extracted", f"{stem}.csv")
    if not os.path.exists(feat_csv):
        if logs is not None:
            logs.append(f"[SimBA] Align skipped (features_extracted missing) for {stem}")
        return

    try:
        df = pd.read_csv(feat_csv)
        # Preserve leading index/frame column if present
        first_col = df.columns[0] if len(df.columns) else None
        keep_first = bool(first_col) and (str(first_col).strip() == "" or str(first_col).startswith("Unnamed") or str(first_col).lower() == "frame")
        # Add missing columns as zeros
        for c in feats:
            if c not in df.columns:
                df[c] = 0
        # Keep expected columns in correct order, but never drop the first column
        if keep_first and first_col not in feats:
            df = df[[first_col] + feats]
        else:
            df = df[feats]
        df.to_csv(feat_csv, index=False)
        if logs is not None:
            logs.append(f"[SimBA] Aligned features to model for {stem} ({len(feats)} cols)")
    except Exception:
        return


def _align_features_to_expected_list(
    project_dir: str,
    stem: str,
    expected_cols: List[str],
    *,
    logs: Optional[list] = None,
) -> None:
    """
    Align features_extracted to a provided expected column list.
    Useful as a fallback when model feature names are unavailable.
    """
    if not expected_cols:
        return
    feat_csv = os.path.join(project_dir, "csv", "features_extracted", f"{stem}.csv")
    if not os.path.exists(feat_csv):
        return
    try:
        import pandas as pd
        df = pd.read_csv(feat_csv)
        first_col = df.columns[0] if len(df.columns) else None
        keep_first = bool(first_col) and (str(first_col).strip() == "" or str(first_col).startswith("Unnamed") or str(first_col).lower() == "frame")
        for c in expected_cols:
            if c not in df.columns:
                df[c] = 0
        # Keep expected columns in order, but never drop the first column
        if keep_first and first_col not in expected_cols:
            df = df[[first_col] + expected_cols]
        else:
            df = df[expected_cols]
        df.to_csv(feat_csv, index=False)
        if logs is not None:
            logs.append(f"[SimBA] Aligned features to expected list for {stem} ({len(expected_cols)} cols)")
    except Exception:
        return


def _add_dummy_index_column(
    project_dir: str,
    stem: str,
    *,
    col_name: str = "frame",
    logs: Optional[list] = None,
) -> None:
    """
    Add a leading dummy index column to protect against readers that use index_col=0.
    """
    feat_csv = os.path.join(project_dir, "csv", "features_extracted", f"{stem}.csv")
    if not os.path.exists(feat_csv):
        return
    try:
        import pandas as pd
        df = pd.read_csv(feat_csv)
        if col_name in df.columns:
            return
        df.insert(0, col_name, range(len(df)))
        df.to_csv(feat_csv, index=False)
        if logs is not None:
            logs.append(f"[SimBA] Added dummy index column: {col_name}")
    except Exception:
        return


def _stabilize_constant_features(
    project_dir: str,
    stem: str,
    *,
    logs: Optional[list] = None,
    epsilon: float = 1e-8,
) -> None:
    """
    Prevent SimBA from dropping constant features by adding a tiny epsilon
    to constant numeric columns (keeps feature counts stable for inference).
    """
    feat_csv = os.path.join(project_dir, "csv", "features_extracted", f"{stem}.csv")
    if not os.path.exists(feat_csv):
        return
    try:
        import pandas as pd
        df = pd.read_csv(feat_csv)
        const_cols = []
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            if df[c].nunique(dropna=True) <= 1:
                const_cols.append(c)
        if not const_cols:
            return
        # Nudge the first row to avoid "constant column" dropping.
        # Use integer-safe nudge for low-probability counters to survive int casts.
        for c in const_cols:
            try:
                if str(c).startswith("Low_prob_detections"):
                    df.loc[df.index[0], c] = 1
                else:
                    df.loc[df.index[0], c] = float(df.loc[df.index[0], c]) + epsilon
            except Exception:
                continue
        df.to_csv(feat_csv, index=False)
        if logs is not None:
            logs.append(f"[SimBA] Stabilized constant features: {const_cols}")
    except Exception:
        return




def _ensure_features_extracted_name(project_dir: str, stem_from: str, stem_to: str) -> None:
    """
    SimBA expects features_extracted to match the pose CSV stem.
    If only the input video stem exists, copy it to the pose stem.
    """
    if stem_from == stem_to:
        return
    feats_dir = os.path.join(project_dir, "csv", "features_extracted")
    src = os.path.join(feats_dir, f"{stem_from}.csv")
    dst = os.path.join(feats_dir, f"{stem_to}.csv")
    if os.path.exists(src) and not os.path.exists(dst):
        try:
            import shutil
            shutil.copy(src, dst)
        except Exception:
            pass


def run_simba_pipeline(
    *,
    config_path: str,
    pose_csv: str,
    input_video: str,
    logs: list,
    out_dir: str,
    force: bool = False,
    kin_csv: Optional[str] = None,
    augment_with_kinematics: bool = False,
    interpolation_setting: Optional[str] = None,
    smoothing_setting: Optional[str] = None,
    smoothing_time_ms: Optional[int] = None,
    px_per_mm: Optional[float] = None,
    render_video: bool = True,
    allow_external: bool = True,
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
        # Expected features_extracted file for this stem
        try:
            from pathlib import Path as _P
            _stem = _P(pose_csv).stem
            _feat = _P(project_dir) / "csv" / "features_extracted" / f"{_stem}.csv"
        except Exception as _e:
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
        if allow_external:
            return _run_simba_pipeline_external(
                config_path=config_path,
                pose_csv=pose_csv,
                input_video=input_video,
                out_dir=out_dir,
                force=force,
                kin_csv=kin_csv,
                augment_with_kinematics=augment_with_kinematics,
                interpolation_setting=interpolation_setting,
                smoothing_setting=smoothing_setting,
                smoothing_time_ms=smoothing_time_ms,
                px_per_mm=px_per_mm,
                render_video=render_video,
                conda_env="simba_fast",
            )
        raise RuntimeError(
            "SimBA import failed in the active environment. "
            f"Import error: {e}"
        ) from e

    project_dir = _resolve_project_dir(config_path)
    msg = [
        f"[SIMBA DEBUG] config_path={config_path}",
        f"[SIMBA DEBUG] project_dir={project_dir}",
        f"[SIMBA DEBUG] pose_csv={pose_csv}",
    ]
    for line in msg:
        print(line, flush=True)
        try:
            logs.append(line)
        except Exception:
            pass
    try:
        _stem = Path(pose_csv).stem
        _feat = Path(project_dir) / "csv" / "features_extracted" / f"{_stem}.csv"
        line1 = f"[SIMBA DEBUG] features_extracted_expected={_feat}"
        line2 = f"[SIMBA DEBUG] features_extracted_exists={_feat.exists()}"
        print(line1, flush=True)
        print(line2, flush=True)
        try:
            logs.append(line1); logs.append(line2)
        except Exception:
            pass
    except Exception as _e:
        line = f"[SIMBA DEBUG] feature path check failed: {_e}"
        print(line, flush=True)
        try:
            logs.append(line)
        except Exception:
            pass
    stem = Path(input_video).stem
    pose_stem = Path(pose_csv).stem
    stem_pose = Path(pose_csv).stem

    _copy_video(input_video, project_dir)
    _ensure_project_bp_names(project_dir, pose_csv)
    # Ensure video_info includes both video stem and pose stem (e.g., *_filtered)
    _ensure_video_info_csv(project_dir, input_video, config_path, video_name_override=stem)
    _ensure_video_info_csv(project_dir, input_video, config_path, video_name_override=pose_stem)
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
    _ensure_features_extracted_name(project_dir, stem_from=stem, stem_to=pose_stem)

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

    def _augment_features_with_kinematics(
        *, project_dir: str, stem: str, kin_csv: str
    ) -> Optional[str]:
        if not kin_csv or not os.path.exists(kin_csv):
            return None
        feats_path = os.path.join(project_dir, "csv", "features_extracted", f"{stem}.csv")
        if not os.path.exists(feats_path):
            return None
        try:
            import pandas as pd
            feats = pd.read_csv(feats_path)
            kin = pd.read_csv(kin_csv)

            # pick numeric kin columns (exclude frame)
            kin_cols = []
            for c in kin.columns:
                if c == "frame":
                    continue
                if pd.api.types.is_numeric_dtype(kin[c]):
                    kin_cols.append(c)
            if not kin_cols:
                return None

            kin = kin[["frame"] + kin_cols] if "frame" in kin.columns else kin[kin_cols]
            # prefix to avoid collisions with SimBA feature names
            kin = kin.rename(columns={c: f"kin_{c}" for c in kin_cols})

            if "frame" in feats.columns and "frame" in kin.columns:
                merged = feats.merge(kin, on="frame", how="left")
            else:
                # fallback: align by index (truncate to shortest)
                n = min(len(feats), len(kin))
                merged = feats.iloc[:n].copy()
                kin = kin.iloc[:n].reset_index(drop=True)
                merged = merged.reset_index(drop=True)
                merged = pd.concat([merged, kin.drop(columns=["frame"], errors="ignore")], axis=1)

            merged.to_csv(feats_path, index=False)
            return feats_path
        except Exception as e:
            logs.append(f"[SimBA] Feature augmentation failed: {e}")
            return None

    # Decide if kinematics augmentation should be used based on model features
    try:
        cfg = configparser.ConfigParser()
        cfg.read(config_path, encoding="utf-8")
        model_paths = []
        if cfg.has_section("SML settings"):
            for key, val in cfg["SML settings"].items():
                if key.startswith("model_path_"):
                    model_paths.append(val)
        expects_kin = False
        try:
            import joblib
            for mp in model_paths:
                if mp and os.path.exists(mp):
                    clf = joblib.load(mp)
                    feats = getattr(clf, "feature_names_in_", None)
                    if feats and any(str(f).startswith("kin_") for f in feats):
                        expects_kin = True
                        break
        except Exception:
            pass
        if augment_with_kinematics and not expects_kin:
            augment_with_kinematics = False
            logs.append("[SimBA] Kinematics augmentation disabled (model expects original features).")
    except Exception:
        pass

    logs.append("[SimBA] Feature extraction")
    feature_extraction_runner(config_path=config_path)
    _ensure_features_extracted_name(project_dir, stem_from=stem, stem_to=stem_pose)
    _ensure_features_extracted_name(project_dir, stem_from=stem_pose, stem_to=stem)

    if kin_csv and augment_with_kinematics:
        aug = _augment_features_with_kinematics(
            project_dir=project_dir,
            stem=stem,
            kin_csv=kin_csv,
        )
        if aug:
            logs.append(f"[SimBA] Augmented features with kinematics: {aug}")
        else:
            logs.append("[SimBA] Kinematics augmentation skipped (missing files or no numeric columns).")
    elif kin_csv and not augment_with_kinematics:
        logs.append("[SimBA] Kinematics augmentation disabled (model expects original features).")

    logs.append("[SimBA] Inference (preconfigured models)")
    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")
    model_paths = []
    if cfg.has_section("SML settings"):
        for key, val in cfg["SML settings"].items():
            if key.startswith("model_path_"):
                model_paths.append(val)
    # Avoid constant-column dropping inside SimBA
    _stabilize_constant_features(project_dir, stem_pose, logs=logs)
    _stabilize_constant_features(project_dir, stem, logs=logs)

    # Build an expected feature list from the current pose-stem features_extracted
    expected_cols = []
    try:
        import pandas as pd
        expected_path = os.path.join(project_dir, "csv", "features_extracted", f"{stem_pose}.csv")
        if os.path.exists(expected_path):
            expected_cols = list(pd.read_csv(expected_path, nrows=1).columns)
    except Exception:
        expected_cols = []

    # Align to expected list first (fallback), then try model-based alignment
    if expected_cols:
        _align_features_to_expected_list(project_dir, stem_pose, expected_cols, logs=logs)
        _align_features_to_expected_list(project_dir, stem, expected_cols, logs=logs)
    _align_features_to_model(project_dir, stem_pose, model_paths, logs=logs)
    _align_features_to_model(project_dir, stem, model_paths, logs=logs)
    try:
        InferenceBatch(config_path=config_path).run()
    except Exception as e:
        msg = str(e)
        if "FeatureNumberMismatchError" in msg and "contains 28 features" in msg:
            logs.append("[SimBA] Retrying inference with dummy index column to avoid index_col=0 drops.")
            _add_dummy_index_column(project_dir, stem_pose, logs=logs)
            _add_dummy_index_column(project_dir, stem, logs=logs)
            InferenceBatch(config_path=config_path).run()
        else:
            raise

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
