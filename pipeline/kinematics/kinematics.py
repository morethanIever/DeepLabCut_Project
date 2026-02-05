# pipeline/kinematics.py
import os
import pandas as pd
import numpy as np
from pipeline.cache_utils import ensure_dirs, cached_kin_path

def get_xy(df, body, coord):
    mask = (
        (df.columns.get_level_values(1) == body)
        & (df.columns.get_level_values(2) == coord)
    )
    cols = df.loc[:, mask]

    if cols.shape[1] == 0:
        raise RuntimeError(
            f"[KIN] Bodypart '{body}' with coord '{coord}' not found.\n"
            f"Available bodyparts: {sorted(set(df.columns.get_level_values(1)))}"
        )

    return cols.iloc[:, 0].to_numpy()


def compute_kinematics(
    pose_csv: str,
    video_path: str,
    logs: list,
    *,
    force: bool = False,
    out_dir: str = "outputs",
    cache_key: str | None = None,
) -> str:
    """
    Compute kinematics with caching.
    """
    ensure_dirs(out_dir)
    kin_cache = cached_kin_path(video_path, out_dir, cache_key)

    # ----------------------------
    # Cache logic
    # ----------------------------
    if (not force) and os.path.exists(kin_cache):
        logs.append(f"[KIN] Using cached kinematics: {kin_cache}")
        return kin_cache

    logs.append("[KIN] Computing kinematics (fast)...")

    df = pd.read_csv(pose_csv, header=[0, 1, 2])

    # ---- Adjust bodypart names to your project ----
    body_nose = "nose"
    body_l_ear = "leftEar"     
    body_r_ear = "rightEar"   
    body_spine = "spineUpper"
    body_spine_lower = "spineLower"  
    body_l_shoulder = "leftShoulder"
    body_r_shoulder = "rightShoulder"
    #body_l_wrist = "leftWrist"
    #body_r_wrist = "rightWrist"
    body_tail_base = "tailBase"


    def col(body, coord):
        return (
            (df.columns.get_level_values(1) == body)
            & (df.columns.get_level_values(2) == coord)
        )

   
    le_x = get_xy(df, body_l_ear, "x")
    le_y = get_xy(df, body_l_ear, "y")
    nose_x = get_xy(df, body_nose, "x")
    nose_y = get_xy(df, body_nose, "y")
    re_x = get_xy(df, body_r_ear, "x")
    re_y = get_xy(df, body_r_ear, "y")
    
    sp_x = get_xy(df, body_spine, "x")
    sp_y = get_xy(df, body_spine, "y")
    spL_x = get_xy(df, body_spine_lower, "x")
    spL_y = get_xy(df, body_spine_lower, "y")
    lShoulder_x = get_xy(df, body_l_shoulder, "x")
    lShoulder_y = get_xy(df, body_l_shoulder, "y")
    rShoulder_x = get_xy(df, body_r_shoulder, "x")
    rShoulder_y = get_xy(df, body_r_shoulder, "y")
    #lWrist_x = get_xy(df, body_l_wrist, "x")
    #lWrist_y = get_xy(df, body_l_wrist, "y")
    #rWrist_x = get_xy(df, body_r_wrist, "x")
    #rWrist_y = get_xy(df, body_r_wrist, "y")
    tailBase_x = get_xy(df, body_tail_base, "x")
    tailBase_y = get_xy(df, body_tail_base, "y")
    

    fps = 30.0
    window = 5
    
    # Smooth coordinates to remove jitter
    sp_x_series = pd.Series(sp_x).rolling(window, center=True).mean().bfill().ffill()
    sp_y_series = pd.Series(sp_y).rolling(window, center=True).mean().bfill().ffill()
    
    sp_x_smooth = sp_x_series.values
    sp_y_smooth = sp_y_series.values

    # Calculate velocity components
    dx = np.diff(sp_x_smooth, prepend=sp_x_smooth[0])
    dy = np.diff(sp_y_smooth, prepend=sp_y_smooth[0])
    
    # Calculate magnitude (Speed)
    speed = np.sqrt(dx**2 + dy**2) * fps
    ear_mid_x = (le_x + re_x) / 2.0
    ear_mid_y = (le_y + re_y) / 2.0
    head_angle_deg = np.degrees(
        np.arctan2(nose_y - ear_mid_y, nose_x - ear_mid_x)
    )
    dtheta = np.abs(np.diff(head_angle_deg, prepend=head_angle_deg[0]))

    # ------------------------------------
    # Movement direction (velocity vector)
    # ------------------------------------
    vx = np.diff(sp_x, prepend=sp_x[0])
    vy = np.diff(sp_y, prepend=sp_y[0])

    v_norm = np.sqrt(vx**2 + vy**2) + 1e-6
    vx_n = vx / v_norm
    vy_n = vy / v_norm

    # Angle change between consecutive velocity vectors
    dot = vx_n * np.roll(vx_n, 1) + vy_n * np.roll(vy_n, 1)
    dot = np.clip(dot, -1, 1)

    turning_angle_deg = np.degrees(np.arccos(dot))
    turning_angle_deg[0] = 0
    
    # head_spine_angle
    head_vec_x = nose_x - sp_x
    head_vec_y = nose_y - sp_y

    spine_vec_x = sp_x - spL_x
    spine_vec_y = sp_y - spL_y

    dot = head_vec_x * spine_vec_x + head_vec_y * spine_vec_y
    mag = (
        np.sqrt(head_vec_x**2 + head_vec_y**2)
        * np.sqrt(spine_vec_x**2 + spine_vec_y**2)
        + 1e-6
    )
    head_spine_angle = np.degrees(np.arccos(np.clip(dot / mag, -1, 1)))

    n = len(sp_x)

    out = pd.DataFrame({
        "frame": np.arange(n),
        "spine_x": sp_x,
        "spine_y": sp_y,
        "speed_px_s": speed,
        "head_angle_deg": head_angle_deg,
        "nose_x": nose_x,
        "nose_y": nose_y,
        "spine_lower_x": spL_x,
        "spine_lower_y": spL_y,
        "left_shoulder_x": lShoulder_x,
        "left_shoulder_y": lShoulder_y,
        "right_shoulder_x": rShoulder_x,
        "right_shoulder_y": rShoulder_y,
        #"left_wrist_x": lWrist_x,
        #"left_wrist_y": lWrist_y,
        #"right_wrist_x": rWrist_x,
        #"right_wrist_y": rWrist_y,
        "tailbase_x": tailBase_x,
        "tailbase_y": tailBase_y,
        "turning_rate_deg":dtheta,
        "move_turn_angle_deg": turning_angle_deg,
        "head_spine_angle_deg": head_spine_angle
    })

    out.to_csv(kin_cache, index=False)
    logs.append(f"[KIN] Kinematics cached at {kin_cache}")

    return kin_cache
