# pipeline/ML/ml_features.py
import os
import pandas as pd
import numpy as np

from pipeline.cache_utils import ensure_dirs, cached_ml_features


def extract_ml_features(
    kin_csv: str,
    video_path: str,
    *,
    force: bool = False,
    smooth_window: int = 15,
    std_window: int = 15,
    lags: list[int] = None,
    include_acceleration: bool = False,
    fps: float | None = None,          # if provided, scale coord velocities to px/s
    include_coord_features: bool = True,
    out_dir: str = "outputs",
    cache_key: str | None = None,
) -> str:
    """
    RandomForest-friendly ML features.
    - Adds frame column
    - Removes redundant transforms
    - Clean smoothing/stats order
    - Adds coordinate-based features (nose/head/body motion) if enabled
    """

    ensure_dirs(out_dir)
    out_path = cached_ml_features(video_path, out_dir, cache_key)

    if (not force) and os.path.exists(out_path):
        print(f"[ML] Using cached features: {out_path}")
        return out_path

    if lags is None:
        lags = [1, 2, 3, 5]

    df = pd.read_csv(kin_csv)

    # If required columns are missing, skip ML features and save a minimal file.
    required = [
        "speed_px_s",
        "turning_rate_deg",
        "move_turn_angle_deg",
        "spine_x", "spine_y",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        frame = df["frame"].astype(int) if "frame" in df.columns else pd.Series(np.arange(len(df)), name="frame")
        minimal = pd.DataFrame({"frame": frame})
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        minimal.to_csv(out_path, index=False)
        print(f"[ML] Skipped features (missing columns {missing}). Saved minimal file: {out_path}")
        return out_path

    def get_dist(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def diff_series(s: pd.Series) -> pd.Series:
        # robust numeric diff
        return s.astype(float).diff().fillna(0.0)

    n = len(df)
    frame = df["frame"].astype(int) if "frame" in df.columns else pd.Series(np.arange(n, dtype=int), name="frame")

    # numeric views (avoid dtype surprises)
    nose_x = df["nose_x"].astype(float) if "nose_x" in df.columns else pd.Series(np.zeros(n))
    nose_y = df["nose_y"].astype(float) if "nose_y" in df.columns else pd.Series(np.zeros(n))
    spine_x = df["spine_x"].astype(float)
    spine_y = df["spine_y"].astype(float)
    spineL_x = df["spine_lower_x"].astype(float) if "spine_lower_x" in df.columns else pd.Series(np.zeros(n))
    spineL_y = df["spine_lower_y"].astype(float) if "spine_lower_y" in df.columns else pd.Series(np.zeros(n))
    lsh_x = df["left_shoulder_x"].astype(float) if "left_shoulder_x" in df.columns else pd.Series(np.zeros(n))
    lsh_y = df["left_shoulder_y"].astype(float) if "left_shoulder_y" in df.columns else pd.Series(np.zeros(n))
    rsh_x = df["right_shoulder_x"].astype(float) if "right_shoulder_x" in df.columns else pd.Series(np.zeros(n))
    rsh_y = df["right_shoulder_y"].astype(float) if "right_shoulder_y" in df.columns else pd.Series(np.zeros(n))

    scale = float(fps) if fps is not None else 1.0  # frame-step -> per-second if fps given

    # --- base features ---
    feats = pd.DataFrame(
        {
            "frame": frame,
            "speed": df["speed_px_s"].astype(float),
            "turn_rate": df["turning_rate_deg"].astype(float),
            "move_turn": df["move_turn_angle_deg"].astype(float),
            "body_stretch": np.hypot(nose_x - spineL_x, nose_y - spineL_y),
            "shoulder_width": get_dist(lsh_x, lsh_y, rsh_x, rsh_y),
            "spine_len": np.hypot(spine_x - spineL_x, spine_y - spineL_y),
            "head_spine_angle": df["head_spine_angle_deg"].astype(float) if "head_spine_angle_deg" in df.columns else 0.0,
        }
    )

    # Optional acceleration from speed
    if include_acceleration:
        accel = feats["speed"].diff().fillna(0.0)
        if fps is not None:
            accel = accel * scale  # px/s^2 (approx)
        feats["acceleration"] = accel

    # --- coordinate-based features (RF-friendly) ---
    if include_coord_features:
        # Per-point velocities (frame-to-frame)
        nose_vx = diff_series(nose_x) * scale
        nose_vy = diff_series(nose_y) * scale
        nose_v = np.hypot(nose_vx, nose_vy)

        spine_vx = diff_series(spine_x) * scale
        spine_vy = diff_series(spine_y) * scale
        spine_v = np.hypot(spine_vx, spine_vy)

        spineL_vx = diff_series(spineL_x) * scale
        spineL_vy = diff_series(spineL_y) * scale
        spineL_v = np.hypot(spineL_vx, spineL_vy)

        # Relative geometry: nose relative to spine
        dx_ns = nose_x - spine_x
        dy_ns = nose_y - spine_y
        nose_spine_dist = np.hypot(dx_ns, dy_ns)

        # Angle of vector spine -> nose (in degrees, wrapped -180..180)
        nose_spine_angle = np.degrees(np.arctan2(dy_ns, dx_ns))

        # Angular change rates (head angle dynamics)
        head_ang_vel = diff_series(feats["head_spine_angle"]) * scale
        turn_rate_vel = diff_series(feats["turn_rate"]) * scale  # jerk-ish in turning

        # “head vs body” activity ratio (often helps sniffing/grooming vs moving)
        # Add epsilon to avoid div by zero
        eps = 1e-6
        nose_to_spine_speed_ratio = nose_v / (spineL_v + eps)

        feats["nose_v"] = nose_v
        feats["nose_vx"] = nose_vx
        feats["nose_vy"] = nose_vy
        feats["spine_v"] = spine_v
        feats["spine_lower_v"] = spineL_v
        feats["nose_spine_dist"] = nose_spine_dist
        feats["nose_spine_angle"] = nose_spine_angle
        feats["head_ang_vel"] = head_ang_vel
        feats["turn_rate_change"] = turn_rate_vel
        feats["nose_to_body_speed_ratio"] = nose_to_spine_speed_ratio

    # --- smoothing (do NOT smooth frame) ---
    smooth_cols = [c for c in feats.columns if c != "frame"]
    smoothed = feats.copy()
    if smooth_window and smooth_window > 1:
        smoothed[smooth_cols] = feats[smooth_cols].rolling(window=smooth_window, center=True).mean()

    # --- window stats from ORIGINAL (not already-smoothed) ---
    # keep these as volatility signals; don't resmooth them.
    if std_window and std_window > 1:
        smoothed["speed_std"] = feats["speed"].rolling(window=std_window, center=True).std()
        smoothed["angle_std"] = feats["head_spine_angle"].rolling(window=std_window, center=True).std()

        if include_coord_features:
            smoothed["nose_v_std"] = feats["nose_v"].rolling(window=std_window, center=True).std()
            smoothed["head_ang_vel_std"] = feats["head_ang_vel"].rolling(window=std_window, center=True).std()
    else:
        smoothed["speed_std"] = 0.0
        smoothed["angle_std"] = 0.0
        if include_coord_features:
            smoothed["nose_v_std"] = 0.0
            smoothed["head_ang_vel_std"] = 0.0

    smoothed = smoothed.fillna(0.0)

    # --- lags (sparse) ---
    final_feats = smoothed.copy()
    lag_cols = [c for c in smoothed.columns if c != "frame"]

    for col in lag_cols:
        for l in lags:
            l = int(l)
            if l <= 0:
                continue
            final_feats[f"{col}_lag_{l}"] = smoothed[col].shift(l)

    final_feats = final_feats.fillna(0.0)

    # save
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    final_feats.to_csv(out_path, index=False)
    print(f"[ML] Features saved to {out_path}")
    return out_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m pipeline.ML.ml_features <kin_csv> <video_path>")
        sys.exit(1)

    extract_ml_features(sys.argv[1], sys.argv[2], force=True)
