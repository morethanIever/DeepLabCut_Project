# pipeline/pose_dlc.py
import os
from pathlib import Path
import pandas as pd

from pipeline.cache_utils import ensure_dirs, cached_pose_path


def run_deeplabcut_pose(video_path: str, logs: list, *, force: bool = False) -> str:
    """
    Run DeepLabCut analysis and return the produced pose CSV path.
    If DLC doesn't export CSV (only H5), convert H5 -> CSV and cache it.
    """
    ensure_dirs()

    pose_cache = cached_pose_path(video_path)

    # Cache hit (unless forced)
    if (not force) and os.path.exists(pose_cache):
        logs.append("[POSE] Cached pose found. Skipping DeepLabCut.")
        return pose_cache

    logs.append("[POSE] No cache found. Running DeepLabCut (slow)...")

    CONFIG_PATH = r"C:\Users\leelab\Desktop\TestBehaviour-Eunhye-2025-12-29\config.yaml"

    try:
        import deeplabcut
    except Exception as e:
        raise RuntimeError(f"DeepLabCut import failed. Is it installed in this env? {e}")

    video_p = Path(video_path)
    video_dir = video_p.parent
    stem = video_p.stem

    # Run DLC
    logs.append(f"[DLC] analyze_videos started (destfolder={video_dir})...")
    cwd = os.getcwd()
    try:
        os.chdir(str(video_dir))
        deeplabcut.analyze_videos(
            CONFIG_PATH,
            [str(video_p)],
            save_as_csv=True,
            batchsize=16,
            destfolder=str(video_dir),
        )
    finally:
        os.chdir(cwd)

    # Filter predictions (optional)
    try:
        logs.append("[DLC] filterpredictions started...")
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

    df = pd.read_hdf(str(src_h5))  # DLC stores a DataFrame inside H5
    df.to_csv(pose_cache)
    logs.append(f"[POSE] Pose cached at {pose_cache} (from H5)")

    return pose_cache
