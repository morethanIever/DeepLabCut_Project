# pipeline/pose_dlc.py
import os
import glob
from pipeline.cache_utils import ensure_dirs, cached_pose_path


def run_deeplabcut_pose(video_path: str,
    logs: list,
    *,
    force: bool = False,) -> str:
    """
    Run DeepLabCut analysis and return the produced CSV path.
    You MUST set CONFIG_PATH to your DLC project's config.yaml.
    """
    ensure_dirs()

    pose_cache = cached_pose_path(video_path)

    # ✅ 캐시 존재하면 DLC 스킵
    if os.path.exists(pose_cache):
        logs.append("[POSE] Cached pose found. Skipping DeepLabCut.")
        return pose_cache

    logs.append("[POSE] No cache found. Running DeepLabCut (slow)...")
    
    # TODO: change this to YOUR config.yaml path
    CONFIG_PATH = r""

    try:
        import deeplabcut
    except Exception as e:
        raise RuntimeError(f"DeepLabCut import failed. Is it installed in this env? {e}")

    logs.append("[DLC] analyze_videos started...")
    deeplabcut.analyze_videos(CONFIG_PATH, [video_path], save_as_csv=True)

    # Optional: filter predictions (helps jitter)
    try:
        logs.append("[DLC] filterpredictions started...")
        deeplabcut.filterpredictions(CONFIG_PATH, [video_path])
    except Exception as e:
        logs.append(f"[WARN] filterpredictions skipped/failed: {e}")

    # DLC saves output next to the video (usually)
    # Find newest CSV that matches the video basename
    base = os.path.splitext(video_path)[0]
    candidates = glob.glob(base + "*filtered.csv") + glob.glob(base + "*DLC*.csv")

    if not candidates:
        raise RuntimeError(
            "Could not find DLC output CSV next to the video. "
            "Check DLC output settings and paths."
        )

    # pick most recently modified
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    src_csv = candidates[0]
    
    os.replace(src_csv, pose_cache)
    logs.append(f"[POSE] Pose cached at {pose_cache}")

    return pose_cache