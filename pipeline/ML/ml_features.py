import os
import pandas as pd
import numpy as np
from pipeline.cache_utils import ensure_dirs, cached_ml_features

# -----------------------------------
# Feature extraction
# -----------------------------------
def extract_ml_features(kin_csv: str, video_path: str, *, force=False) -> str:
    ensure_dirs()
    out_path = cached_ml_features(video_path)

    if (not force) and os.path.exists(out_path):
        print(f"[ML] Using cached features: {out_path}")
        return out_path

    print("[ML] Extracting ML features...")

    df = pd.read_csv(kin_csv)

    def get_dist(x1, y1, x2, y2):
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    required = [
        "speed_px_s",
        "turning_rate_deg",
        "move_turn_angle_deg",
        "nose_x",
        "nose_y",
        "spine_x",
        "spine_y",
        "left_shoulder_x",
        "left_shoulder_y",
        "right_shoulder_x",
        "right_shoulder_y",
        "spine_lower_x",
        "spine_lower_y",
        "head_spine_angle_deg",
    ]

    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"[ML] Missing column: {c}")

    feats = pd.DataFrame({
        "speed": df["speed_px_s"],
        "acceleration": df["speed_px_s"].diff().fillna(0),
        "log_speed": np.log1p(df["speed_px_s"]),
        "body_stretch": np.sqrt((df["nose_x"]-df["spine_lower_x"])**2 + (df["nose_y"]-df["spine_lower_y"])**2),
        "shoulder_width": get_dist(df["left_shoulder_x"], df["left_shoulder_y"], df["right_shoulder_x"], df["right_shoulder_y"]),
        "turn_rate": df["turning_rate_deg"],
        "angular_velocity": df["turning_rate_deg"].abs(),
        "move_turn": df["move_turn_angle_deg"],
        "spine_len": np.hypot(
            df["spine_x"] - df["spine_lower_x"],
            df["spine_y"] - df["spine_lower_y"],
        ),
        "head_spine_angle": df["head_spine_angle_deg"],
        "vel_std": df["speed_px_s"].rolling(window=30, center=True).std(),
        "angle_std": df["head_spine_angle_deg"].rolling(window=30, center=True).std().fillna(0),
    })

    # temporal smoothing
    feats = feats.rolling(5, center=True).mean().fillna(0)

    feats.to_csv(out_path, index=False)
    print(f"[ML] Features saved to {out_path}")

    return out_path


# -----------------------------------
# CLI entry
# -----------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m pipeline.ML.ml_features <kin_csv> <video_path>")
        sys.exit(1)

    extract_ml_features(sys.argv[1], sys.argv[2], force=True)
