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

    required = [
        "speed_px_s",
        "turning_rate_deg",
        "move_turn_angle_deg",
        "spine_x",
        "spine_y",
        "spine_lower_x",
        "spine_lower_y",
        "head_spine_angle_deg",
    ]

    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"[ML] Missing column: {c}")

    feats = pd.DataFrame({
        "speed": df["speed_px_s"],
        "turn_rate": df["turning_rate_deg"],
        "move_turn": df["move_turn_angle_deg"],
        "spine_len": np.hypot(
            df["spine_x"] - df["spine_lower_x"],
            df["spine_y"] - df["spine_lower_y"],
        ),
        "head_spine_angle": df["head_spine_angle_deg"],
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
        print("Usage: python ml_features.py <kin_csv> <video_path>")
        sys.exit(1)

    extract_ml_features(sys.argv[1], sys.argv[2], force=True)
