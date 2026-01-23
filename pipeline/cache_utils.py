import os
import hashlib

# ----------------------------
# Directory setup
# ----------------------------
def ensure_dirs():
    os.makedirs("outputs/poses", exist_ok=True)
    os.makedirs("outputs/kinematics", exist_ok=True)
    os.makedirs("outputs/behavior", exist_ok=True)
    os.makedirs("outputs/videos", exist_ok=True)
    os.makedirs("outputs/nop", exist_ok=True)
    os.makedirs("outputs/ml", exist_ok=True)
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
def _safe_stem(video_path: str) -> str:
    """
    Stable cache key for the same video content,
    regardless of temp filename or path.
    """
    return file_content_hash(video_path)

# ----------------------------
# Cached paths
# ----------------------------
def cached_pose_path(video_path: str) -> str:
    return os.path.join("outputs", "poses", _safe_stem(video_path) + "_filtered.csv")

def cached_kin_path(video_path: str) -> str:
    return os.path.join("outputs", "kinematics", _safe_stem(video_path) + "_kinematics.csv")

def cached_beh_path(video_path: str) -> str:
    return os.path.join("outputs", "behavior", _safe_stem(video_path) + "_behavior.csv")

def cached_outvideo_path(video_path: str) -> str:
    return os.path.join("outputs", "videos", _safe_stem(video_path) + "_annotated.mp4")

def cached_turn_path(video_path: str) -> str:
    return os.path.join(
        "outputs",
        "kinematics",
        _safe_stem(video_path) + "_turning_rate.csv"
    )
def cached_nop_path(video_path: str) -> str:
    return os.path.join("outputs", "nop", _safe_stem(video_path) + "nop_summary.csv")

def cached_ml_features(video_path:str) -> str:
    return os.path.join("outputs", "ml", _safe_stem(video_path) + "_ml_features.csv")
