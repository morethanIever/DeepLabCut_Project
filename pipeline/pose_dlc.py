# pipeline/pose_dlc.py
import os
from pathlib import Path
import pandas as pd

from pipeline.cache_utils import ensure_dirs, cached_pose_path

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

def run_deeplabcut_pose(
    video_path: str,
    logs: list,
    *,
    CONFIG_PATH: str,
    force: bool = False,
    out_dir: str = "outputs",
    cache_key: str | None = None,
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
    except Exception as e:
        raise RuntimeError(f"DeepLabCut import failed. Is it installed in this env? {e}")

    video_p = Path(video_path)
    video_dir = video_p.parent
    stem = video_p.stem

    # Resolve engine/shuffle preference (prefer pytorch shuffle when available)
    shuffle_override = None
    if yaml is not None:
        try:
            cfg = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8")) or {}
            engine_pref = str(cfg.get("engine", "")).strip().lower()
            if engine_pref == "pytorch":
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
        except Exception:
            shuffle_override = None

    # Run DLC
    logs.append(f"[DLC] analyze_videos started (destfolder={video_dir})...")
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
