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

    # -------------------------------------------------
    # Pre-computed quantities
    # -------------------------------------------------

    speed = df["speed_px_s"].to_numpy()
    turn_rate = df["turning_rate_deg"].to_numpy()

    # Spine geometry
    spine_len = np.hypot(
        df["spine_x"] - df["spine_lower_x"],
        df["spine_y"] - df["spine_lower_y"],
    )

    spine_mean = spine_len.mean()
    spine_std = spine_len.std() 

    # Tail movement (proxy for vertical posture)
    tail_dx = np.diff(df["tailbase_x"], prepend=df["tailbase_x"].iloc[0])
    tail_dy = np.diff(df["tailbase_y"], prepend=df["tailbase_y"].iloc[0])
    tail_speed = np.sqrt(tail_dx**2 + tail_dy**2)

    # Head–spine angle (grooming / turning cue)
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

    # -------------------------------------------------
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

    turn_angle = (
        df["move_turn_angle_deg"]
        .rolling(window=5, center=True)
        .mean()
        .fillna(0)
    )

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
    rearing_candidate = (
        (tail_speed < np.percentile(tail_speed, 30)) &   # 꼬리 거의 안 움직임
        (spine_len < spine_mean + 0.9* spine_std)        # 상대적 신장
    )

    rearing = enforce_min_duration(rearing_candidate, min_len=15)
    behavior[rearing] = "rearing"
    confidence[rearing] = 0.01 + 0.99 * np.clip(
        (spine_mean - spine_len[rearing]) / spine_std, 0, 1
    )
    # -------------------------------------------------
    # GROOMING
    # -------------------------------------------------
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
