import os
import hashlib

# ----------------------------
# Directory setup
# ----------------------------
def ensure_dirs(base_dir: str = "outputs"):
    os.makedirs(os.path.join(base_dir, "poses"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "kinematics"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "behavior"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "nop"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "ml"), exist_ok=True)
# ----------------------------
# File content hash
# ----------------------------
def file_content_hash(path: str, chunk_size: int = 1 << 20) -> str:
    """
    Compute MD5 hash based on file content (not path).
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

# ----------------------------
# Cache key
# ----------------------------
def _safe_stem(video_path: str, cache_key: str | None = None) -> str:
    """
    Cache key for output naming.
    If cache_key is provided, use it; otherwise fall back to content hash.
    """
    if cache_key:
        return cache_key
    return file_content_hash(video_path)

# ----------------------------
# Cached paths
# ----------------------------
def cached_pose_path(video_path: str, base_dir: str = "outputs", cache_key: str | None = None) -> str:
    return os.path.join(base_dir, "poses", _safe_stem(video_path, cache_key) + "_filtered.csv")

def cached_kin_path(video_path: str, base_dir: str = "outputs", cache_key: str | None = None) -> str:
    return os.path.join(base_dir, "kinematics", _safe_stem(video_path, cache_key) + "_kinematics.csv")

def cached_beh_path(video_path: str, base_dir: str = "outputs", cache_key: str | None = None) -> str:
    return os.path.join(base_dir, "behavior", _safe_stem(video_path, cache_key) + "_behavior.csv")

def cached_outvideo_path(video_path: str, base_dir: str = "outputs", cache_key: str | None = None) -> str:
    return os.path.join(base_dir, "videos", _safe_stem(video_path, cache_key) + "_annotated.mp4")

def cached_turn_path(video_path: str, base_dir: str = "outputs", cache_key: str | None = None) -> str:
    return os.path.join(
        base_dir,
        "kinematics",
        _safe_stem(video_path, cache_key) + "_turning_rate.csv"
    )
def cached_nop_path(video_path: str, base_dir: str = "outputs", cache_key: str | None = None) -> str:
    return os.path.join(base_dir, "nop", _safe_stem(video_path, cache_key) + "nop_summary.csv")

def cached_ml_features(video_path:str, base_dir: str = "outputs", cache_key: str | None = None) -> str:
    return os.path.join(base_dir, "ml", _safe_stem(video_path, cache_key) + "_ml_features.csv")
