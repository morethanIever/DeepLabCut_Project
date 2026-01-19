import os
import numpy as np
import pandas as pd
from pipeline.cache_utils import ensure_dirs, cached_turn_path


def compute_turning_rate(
    kin_csv: str,
    video_path: str,
    logs: list,
    *,
    force: bool = False,
    fps: float = 30.0,
    smooth_window: int = 10,
) -> str:
    """
    Compute turning rate (deg/sec) from spine trajectory.
    """
    ensure_dirs()
    out_path = cached_turn_path(video_path)

    if (not force) and os.path.exists(out_path):
        logs.append(f"[TURN] Using cached turning rate: {out_path}")
        return out_path

    logs.append("[TURN] Computing turning rate...")

    df = pd.read_csv(kin_csv)
    
    

    x = df["spine_x"].to_numpy()
    y = df["spine_y"].to_numpy()

    # smoothing
    x = pd.Series(x).rolling(smooth_window, center=True).mean().to_numpy()
    y = pd.Series(y).rolling(smooth_window, center=True).mean().to_numpy()

    valid = ~np.isnan(x)
    x = x[valid]
    y = y[valid]
    frames = df["frame"].to_numpy()[valid]

    # ... (previous smoothing and valid frame logic) ...

    dx = np.diff(x)
    dy = np.diff(y)
    
    # 1. Calculate step distance (velocity)
    dist = np.sqrt(dx**2 + dy**2)
    
    # 2. Compute heading
    heading = np.arctan2(dy, dx)
    
    # 3. Masking: If movement is less than 1 pixel (jitter), 
    # keep the previous heading instead of calculating a new (noisy) one.
    threshold = 1.0  # adjust based on your tracking quality
    for i in range(1, len(heading)):
        if dist[i] < threshold:
            heading[i] = heading[i-1]

    heading = np.unwrap(heading)
    
    # 4. Calculate change and apply a median filter to squash extreme spikes
    dtheta = np.diff(heading)
    
    # Using a median filter to remove any remaining single-frame outliers
    from scipy.signal import medfilt
    dtheta = medfilt(dtheta, kernel_size=5) 

    turning_rate = np.degrees(dtheta) * fps

    out = pd.DataFrame({
        "frame": frames[-len(turning_rate):],
        "turning_rate_deg_s": turning_rate,
    })

    out.to_csv(out_path, index=False)
    logs.append("[TURN] Turning rate computed.")

    return out_path
