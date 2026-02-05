# pipeline/behavior.py
import os
import numpy as np
import pandas as pd
from pipeline.cache_utils import ensure_dirs, cached_beh_path



# -------------------------------------------------
# Utility: enforce temporal persistence
# -------------------------------------------------
def enforce_min_duration(mask: np.ndarray, min_len: int) -> np.ndarray:
    """
    Keep True only if condition holds for >= min_len consecutive frames
    """
    out = np.zeros_like(mask, dtype=bool)
    count = 0
    for i, v in enumerate(mask):
        if v:
            count += 1
        else:
            count = 0
        if count >= min_len:
            out[i - min_len + 1 : i + 1] = True
    return out


# -------------------------------------------------
# Main behavior classifier
# -------------------------------------------------
def classify_behavior(
    kin_csv: str,
    video_path: str,
    logs: list,
    *,
    force: bool = False,
    out_dir: str = "outputs",
    cache_key: str | None = None,
) -> str:
    """
    Rule-based behavior classification (biologically realistic, top-view).
    """
    ensure_dirs(out_dir)
    beh_cache = cached_beh_path(video_path, out_dir, cache_key)

    # ---------------- Cache ----------------
    if (not force) and os.path.exists(beh_cache):
        logs.append(f"[BEH] Using cached behavior: {beh_cache}")
        return beh_cache

    logs.append("[BEH] Computing behavior (rule-based, realistic)...")

    df = pd.read_csv(kin_csv)
    n = len(df)

    # If required kinematics are missing, skip behavior classification.
    required_cols = ["spine_x", "spine_y", "speed_px_s"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logs.append(f"[BEH] Skipped behavior: missing columns {missing}")
        frames = df["frame"] if "frame" in df.columns else pd.Series(np.arange(n), name="frame")
        out = pd.DataFrame({
            "frame": frames,
            "behavior": ["unknown"] * n,
            "confidence": [0.0] * n,
        })
        out.to_csv(beh_cache, index=False)
        logs.append(f"[BEH] Placeholder behavior saved at {beh_cache}")
        return beh_cache
    # -------------------------------------------------
    # Pre-computed quantities (robust to missing columns)
    # -------------------------------------------------
    
    speed = df["speed_px_s"].to_numpy() if "speed_px_s" in df.columns else np.zeros(n)
    turn_rate = df["turning_rate_deg"].to_numpy() if "turning_rate_deg" in df.columns else np.zeros(n)
    
    has_spine_lower = "spine_lower_x" in df.columns and "spine_lower_y" in df.columns
    has_tail = "tailbase_x" in df.columns and "tailbase_y" in df.columns
    has_nose = "nose_x" in df.columns and "nose_y" in df.columns
    
    if has_spine_lower:
        spine_len = np.hypot(
            df["spine_x"] - df["spine_lower_x"],
            df["spine_y"] - df["spine_lower_y"],
        )
        spine_mean = spine_len.mean()
        spine_std = spine_len.std()
    else:
        spine_len = None
        spine_mean = 0.0
        spine_std = 1.0
    
    if has_tail:
        tail_dx = np.diff(df["tailbase_x"], prepend=df["tailbase_x"].iloc[0])
        tail_dy = np.diff(df["tailbase_y"], prepend=df["tailbase_y"].iloc[0])
        tail_speed = np.sqrt(tail_dx**2 + tail_dy**2)
    else:
        tail_speed = None
    
    if has_nose and has_spine_lower:
        head_vec_x = df["nose_x"] - df["spine_x"]
        head_vec_y = df["nose_y"] - df["spine_y"]
        spine_vec_x = df["spine_x"] - df["spine_lower_x"]
        spine_vec_y = df["spine_y"] - df["spine_lower_y"]
    
        dot = head_vec_x * spine_vec_x + head_vec_y * spine_vec_y
        mag = (
            np.sqrt(head_vec_x**2 + head_vec_y**2)
            * np.sqrt(spine_vec_x**2 + spine_vec_y**2)
            + 1e-6
        )
        head_spine_angle = np.degrees(np.arccos(np.clip(dot / mag, -1, 1)))
    else:
        head_spine_angle = None
    
    # 1️⃣ Base behavior (dominant)
    # -------------------------------------------------
    # -------------------------------------------------
    # Base init
    # -------------------------------------------------
    behavior = np.full(n, "unknown", dtype=object)
    #behavior = []
    confidence = np.full(n, np.nan)
    
    # -------------------------------------------------
    # TURNING 
    # -------------------------------------------------
    speed = df["speed_px_s"].to_numpy()

    dx = np.diff(df["spine_x"], prepend=df["spine_x"].iloc[0])
    dy = np.diff(df["spine_y"], prepend=df["spine_y"].iloc[0])
    disp = np.sqrt(dx**2 + dy**2)

    if "move_turn_angle_deg" in df.columns:
        turn_angle = (
            df["move_turn_angle_deg"]
            .rolling(window=5, center=True)
            .mean()
            .fillna(0)
        )
    else:
        turn_angle = pd.Series(np.zeros(n))

    turn_integral = (
        turn_angle
        .rolling(window=7)
        .sum()
        .fillna(0)
    )

    turn_candidate = (
        (speed > 10) &
        (disp > 6.0) &
        (turn_integral > 60)   # 누적 회전
    )

    turning = enforce_min_duration(turn_candidate, min_len=3)
    behavior[turning] = "turning"
    confidence[turning] = 0.01 + 0.99 * np.clip(
        (turn_integral[turning] - 60) / 60, 0, 1
    )

    # -------------------------------------------------
    # REARING (top-view friendly)
    # -------------------------------------------------
    if tail_speed is not None and spine_len is not None:
        rearing_candidate = (
            (tail_speed < np.percentile(tail_speed, 30)) &
            (spine_len < spine_mean + 0.9 * spine_std)
        )
    
        rearing = enforce_min_duration(rearing_candidate, min_len=15)
        behavior[rearing] = "rearing"
        confidence[rearing] = 0.01 + 0.99 * np.clip(
            (spine_mean - spine_len[rearing]) / spine_std, 0, 1
        )
    # -------------------------------------------------
    # GROOMING
    # -------------------------------------------------
    if head_spine_angle is not None and spine_len is not None:
        grooming_candidate = (
            (speed < 10)
            & (head_spine_angle > 65)
            & (spine_len < spine_mean)
        )

        grooming = enforce_min_duration(grooming_candidate, min_len=15)
        behavior[grooming] = "grooming"
        confidence[grooming] = 0.01 + 0.99 * np.clip(
            (head_spine_angle[grooming] - 65) / 30, 0, 1
        )
    
    # -------------------------------------------------
    # FAST MOVE (맨 마지막)
    # -------------------------------------------------
    fast = (speed > 200) & (behavior == "unknown")
    behavior[fast] = "fast_move"
    confidence[fast] = np.clip((speed[fast] - 200) / 150, 0, 1)
    # -------------------------------------------------
    # REST
    # -------------------------------------------------
    rest = (speed < 15) & (behavior == "unknown")
    behavior[rest] = "rest"    
    confidence[rest] = 0.01 + 0.99 * np.clip(1-speed[rest] / 10, 0, 1)
    
    move_candidate = (
        (speed >= 15) &
        (speed <= 200) &
        (disp > 0.5)
    )

    #MOVE (remaining)
    move = move_candidate & (behavior == "unknown")
    behavior[move] = "move"
    confidence[move] = 0.01 + 0.99 * np.clip(
        1 - np.abs(speed[move] - 40) / 40, 0, 1
    )
    confidence = np.nan_to_num(confidence, nan=0.0)

    # -------------------------------------------------
    # Save
    # -------------------------------------------------
    out = pd.DataFrame({
        "frame": df["frame"],
        "behavior": behavior,
        "confidence": confidence
    })

    out.to_csv(beh_cache, index=False)
    logs.append(f"[BEH] Behavior cached at {beh_cache}")

    # Debug summary
    unique, counts = np.unique(behavior, return_counts=True)
    summary = dict(zip(unique, counts))
    logs.append(f"[BEH] Distribution: {summary}")

    return beh_cache
