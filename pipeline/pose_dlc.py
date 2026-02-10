# pipeline/pose_dlc.py
import os
import subprocess
from pathlib import Path
import pandas as pd
from typing import Optional

from pipeline.cache_utils import ensure_dirs, cached_pose_path

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _get_conda_exe() -> str:
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and os.path.exists(conda_exe):
        return conda_exe
    default = r"C:\Users\leelab\anaconda3\Scripts\conda.exe"
    if os.path.exists(default):
        return default
    raise RuntimeError(
        "conda executable not found. Set CONDA_EXE or install Anaconda/Miniconda."
    )


def _run_dlc_external(
    *,
    action: str,
    config_path: str,
    video_path: str,
    destfolder: str,
    logs: list,
    shuffle: Optional[int] = None,
    batchsize: int = 16,
    save_as_csv: bool = True,
    log_callback: Optional[callable] = None,
) -> None:
    env_name = os.environ.get("DLC_CONDA_ENV", "dlc_gpu")
    conda_exe = _get_conda_exe()
    project_root = str(_get_project_root())

    cmd = [
        conda_exe,
        "run",
        "-n",
        env_name,
        "python",
        "-u",
        "-m",
        "pipeline.dlc_worker",
        "--action",
        action,
        "--config",
        config_path,
        "--video",
        video_path,
        "--destfolder",
        destfolder,
        "--batchsize",
        str(batchsize),
    ]
    if save_as_csv:
        cmd.append("--save-as-csv")
    if shuffle is not None:
        cmd.extend(["--shuffle", str(shuffle)])

    logs.append(f"[DLC] Using external env '{env_name}' via conda run.")
    logs.append(f"[DLC] Command: {' '.join(cmd)}")
    if log_callback:
        log_callback(f"[DLC] Using external env '{env_name}' via conda run.")
        log_callback(f"[DLC] Command: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    output_lines = []
    if proc.stdout:
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            output_lines.append(line)
            logs.append(line)
            if log_callback:
                log_callback(line)
    ret = proc.wait()
    if ret != 0:
        detail = "\n".join(output_lines[-200:]) if output_lines else "(no output captured)"
        raise RuntimeError(
            "External DeepLabCut failed with exit code "
            f"{ret}. Output:\n{detail}"
        )

def run_deeplabcut_pose(
    video_path: str,
    logs: list,
    *,
    CONFIG_PATH: str,
    force: bool = False,
    out_dir: str = "outputs",
    cache_key: Optional[str] = None,
    log_callback: Optional[callable] = None,
) -> str:
    """
    Run DeepLabCut analysis and return the produced pose CSV path.
    If DLC doesn't export CSV (only H5), convert H5 -> CSV and cache it.
    """
    ensure_dirs(out_dir)

    if not CONFIG_PATH or (not os.path.exists(CONFIG_PATH)):
        raise RuntimeError(f"[POSE] DLC config.yaml not found: {CONFIG_PATH}")

    pose_cache = cached_pose_path(video_path, out_dir, cache_key)

    # Cache hit (unless forced)
    if (not force) and os.path.exists(pose_cache):
        logs.append("[POSE] Cached pose found. Skipping DeepLabCut.")
        return pose_cache

    logs.append("[POSE] No cache found. Running DeepLabCut (slow)...")

    try:
        import deeplabcut
        use_external = False
    except Exception:
        deeplabcut = None
        use_external = True

    video_p = Path(video_path)
    video_dir = video_p.parent
    stem = video_p.stem

    # Resolve engine/shuffle preference (prefer pytorch shuffle when available)
    # HARD OVERRIDE: force shuffle 1 to avoid missing-shuffle errors
    shuffle_override = 1
    if yaml is not None:
        try:
            cfg = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8")) or {}
            engine_pref = str(cfg.get("engine", "")).strip().lower()
            project_path = cfg.get("project_path", "")
            if isinstance(project_path, list):
                project_path = " ".join([str(x) for x in project_path if x is not None])
            project_path = str(project_path).strip()
            if project_path:
                dlc_models = Path(project_path) / "dlc-models"
                if not dlc_models.exists() or not any(dlc_models.rglob("*")):
                    raise RuntimeError(
                        f"[POSE] No trained DLC models found under {dlc_models}. "
                        "Run create_training_dataset + train_network first, "
                        "or point to a project with trained weights."
                    )
            task = cfg.get("Task", "")
            date = cfg.get("date", "")
            iteration = cfg.get("iteration", 0)
            meta = None
            if task and date and project_path:
                meta = Path(project_path) / "training-datasets" / f"iteration-{iteration}" / f"UnaugmentedDataSet_{task}{date}" / "metadata.yaml"

            def _pick_shuffle_from_metadata() -> Optional[int]:
                if meta is None or not meta.exists() or yaml is None:
                    return None
                meta_data = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
                shuffles = meta_data.get("shuffles", {}) or {}
                candidates = []
                for name, info in shuffles.items():
                    idx = None
                    if isinstance(info, dict):
                        idx = info.get("index", None)
                    if idx is None:
                        try:
                            idx = int(name)
                        except Exception:
                            idx = None
                    if idx is not None:
                        candidates.append(int(idx))
                if not candidates:
                    return None
                return sorted(candidates)[-1]

            if engine_pref == "pytorch" and meta is not None and meta.exists():
                meta_data = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
                shuffles = meta_data.get("shuffles", {}) or {}
                # pick a pytorch shuffle with the highest index
                candidates = []
                for name, info in shuffles.items():
                    if str(info.get("engine", "")).lower() == "pytorch":
                        candidates.append((int(info.get("index", 0)), name))
                if candidates:
                    candidates.sort(reverse=True)
                    shuffle_override = candidates[0][0]

            if shuffle_override is None:
                cfg_shuffle = cfg.get("shuffle", None)
                best = _pick_shuffle_from_metadata()
                if best is not None:
                    try:
                        cfg_shuffle = int(cfg_shuffle) if cfg_shuffle is not None else None
                    except Exception:
                        cfg_shuffle = None
                    if cfg_shuffle is None or cfg_shuffle != best:
                        shuffle_override = best
                        logs.append(f"[DLC] Overriding shuffle to {best} based on metadata.")
        except Exception:
            shuffle_override = None

    # Run DLC
    logs.append(f"[DLC] analyze_videos started (destfolder={video_dir})...")
    if use_external:
        _run_dlc_external(
            action="analyze",
            config_path=str(Path(CONFIG_PATH).resolve()),
            video_path=str(video_p.resolve()),
            destfolder=str(video_dir.resolve()),
            logs=logs,
            shuffle=shuffle_override,
            log_callback=log_callback,
        )
    else:
        cwd = os.getcwd()
        try:
            os.chdir(str(video_dir))
            try:
                kwargs = dict(
                    save_as_csv=True,
                    batchsize=16,
                    destfolder=str(video_dir),
                )
                if shuffle_override:
                    kwargs["shuffle"] = shuffle_override
                    logs.append(f"[DLC] Using pytorch shuffle {shuffle_override}")
                deeplabcut.analyze_videos(
                    CONFIG_PATH,
                    [str(video_p)],
                    **kwargs,
                )
            except ValueError as e:
                msg = str(e)
                if "Could not find a shuffle" in msg:
                    detail = ""
                    if yaml is not None:
                        try:
                            cfg = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8")) or {}
                            task = cfg.get("Task", "")
                            date = cfg.get("date", "")
                            iteration = cfg.get("iteration", 0)
                            project_path = cfg.get("project_path", "")
                            if isinstance(project_path, list):
                                project_path = " ".join([str(x) for x in project_path if x is not None])
                            project_path = str(project_path).strip()
                            if task and date and project_path:
                                meta = Path(project_path) / "training-datasets" / f"iteration-{iteration}" / f"UnaugmentedDataSet_{task}{date}" / "metadata.yaml"
                                if meta.exists():
                                    detail = f"Found metadata at {meta}, but it has no shuffles. Create a training dataset and train a network first."
                                else:
                                    detail = f"Training metadata not found at {meta}. Create a training dataset and train a network first."
                        except Exception:
                            detail = ""
                    raise RuntimeError(
                        "[POSE] No trained shuffle found for this config. "
                        "Run DLC create_training_dataset + train_network (or point to a trained project). "
                        + (detail or "")
                    ) from e
                raise
        finally:
            os.chdir(cwd)

    # Filter predictions (optional)
    try:
        logs.append("[DLC] filterpredictions started...")
        if use_external:
            _run_dlc_external(
                action="filter",
                config_path=str(Path(CONFIG_PATH).resolve()),
                video_path=str(video_p.resolve()),
                destfolder=str(video_dir.resolve()),
                logs=logs,
                save_as_csv=False,
                log_callback=log_callback,
            )
        else:
            deeplabcut.filterpredictions(
                CONFIG_PATH,
                [str(video_p)],
                destfolder=str(video_dir),
            )
    except Exception as e:
        logs.append(f"[WARN] filterpredictions skipped/failed: {e}")

    # 1) Try find CSV
    csv_candidates = (
        list(video_dir.rglob(f"*{stem}*filtered*.csv"))
        + list(video_dir.rglob(f"*{stem}*DLC*.csv"))
    )

    if csv_candidates:
        csv_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        src_csv = csv_candidates[0]
        os.replace(str(src_csv), pose_cache)
        logs.append(f"[POSE] Pose cached at {pose_cache} (from CSV)")
        return pose_cache

    # 2) No CSV → convert newest H5 to CSV
    h5_candidates = list(video_dir.rglob(f"*{stem}*filtered*.h5"))
    if not h5_candidates:
        h5_candidates = list(video_dir.rglob(f"*{stem}*DLC*.h5"))
    if not h5_candidates:
        h5_candidates = list(video_dir.rglob(f"*{stem}*.h5"))

    if not h5_candidates:
        raise RuntimeError(
            f"Could not find DLC output (CSV or H5) for stem='{stem}'. "
            f"Searched under: {video_dir}"
        )

    h5_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    src_h5 = h5_candidates[0]
    logs.append(f"[POSE] CSV missing. Converting H5 → CSV: {src_h5.name}")

    df = pd.read_hdf(str(src_h5))
    df.to_csv(pose_cache)
    logs.append(f"[POSE] Pose cached at {pose_cache} (from H5)")

    return pose_cache


def extract_outlier_frames(
    *, config_path: str, video_path: str, logs: Optional[list] = None
) -> None:
    logs = logs if logs is not None else []
    try:
        import deeplabcut
        use_external = False
    except Exception:
        deeplabcut = None
        use_external = True

    if use_external:
        _run_dlc_external(
            action="extract_outliers",
            config_path=str(Path(config_path).resolve()),
            video_path=str(Path(video_path).resolve()),
            destfolder=str(Path(video_path).resolve().parent),
            logs=logs,
            save_as_csv=False,
        )
    else:
        deeplabcut.extract_outlier_frames(config=config_path, videos=[video_path])
