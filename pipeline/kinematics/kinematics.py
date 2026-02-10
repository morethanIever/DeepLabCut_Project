# pipeline/kinematics.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Union, List
from pipeline.cache_utils import ensure_dirs, cached_kin_path

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

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


def _load_alias_map() -> dict:
    """
    Load alias map from kinematics.yaml.
    """
    alias_path = Path(__file__).with_name("kinematics.yaml")
    if yaml is None or (not alias_path.exists()):
        return {}
    try:
        data = yaml.safe_load(alias_path.read_text(encoding="utf-8")) or {}
        return data.get("kinematics", {}).get("parts", {}) or {}
    except Exception:
        return {}


def _load_project_mapping(mapping_path: Optional[str]) -> dict:
    """
    Load project-level mapping (json or yaml). Expected format:
    { "nose": "snout", "spine_upper": "spine1", ... }
    """
    if not mapping_path:
        return {}
    p = Path(mapping_path)
    if not p.exists():
        return {}
    try:
        if p.suffix.lower() in {".yml", ".yaml"}:
            if yaml is None:
                return {}
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            return data.get("parts", data)
        if p.suffix.lower() == ".json":
            return pd.read_json(p, typ="series").to_dict()
    except Exception:
        return {}
    return {}


def _resolve_part(bodyparts: List[str], aliases: List[str], preferred: Optional[Union[str, List[str]]]) -> Optional[str]:
    """
    Resolve a bodypart name from available bodyparts.
    """
    lower_map = {bp.lower(): bp for bp in bodyparts}
    if preferred:
        prefs = preferred if isinstance(preferred, list) else [preferred]
        for p in prefs:
            if not isinstance(p, str):
                continue
            if p in bodyparts:
                return p
            if p.lower() in lower_map:
                return lower_map[p.lower()]
    for a in aliases:
        if a in bodyparts:
            return a
        if isinstance(a, str) and a.lower() in lower_map:
            return lower_map[a.lower()]
    return None


def compute_kinematics(
    pose_csv: str,
    video_path: str,
    logs: list,
    *,
    force: bool = False,
    out_dir: str = "outputs",
    cache_key: Optional[str] = None,
    mapping_path: Optional[str] = None,
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
    n = len(df)

    bodyparts = sorted(set(df.columns.get_level_values(1)))
    bodyparts = [bp for bp in bodyparts if isinstance(bp, str) and bp.lower() not in {"bodyparts", "coords", "scorer"}]
    alias_map = _load_alias_map()
    project_map = _load_project_mapping(mapping_path)

    def aliases(key: str) -> List[str]:
        v = alias_map.get(key, [])
        return v if isinstance(v, list) else [v]

    resolved = {
        "nose": _resolve_part(bodyparts, aliases("nose"), project_map.get("nose")),
        "left_ear": _resolve_part(bodyparts, aliases("left_ear"), project_map.get("left_ear")),
        "right_ear": _resolve_part(bodyparts, aliases("right_ear"), project_map.get("right_ear")),
        "neck": _resolve_part(bodyparts, aliases("neck"), project_map.get("neck")),
        "spine_upper": _resolve_part(bodyparts, aliases("spine_upper"), project_map.get("spine_upper")),
        "spine_lower": _resolve_part(bodyparts, aliases("spine_lower"), project_map.get("spine_lower")),
        "left_shoulder": _resolve_part(bodyparts, aliases("left_shoulder"), project_map.get("left_shoulder")),
        "right_shoulder": _resolve_part(bodyparts, aliases("right_shoulder"), project_map.get("right_shoulder")),
        "left_leg": _resolve_part(bodyparts, aliases("left_leg"), project_map.get("left_leg")),
        "right_leg": _resolve_part(bodyparts, aliases("right_leg"), project_map.get("right_leg")),
        "tail_base": _resolve_part(bodyparts, aliases("tail_base"), project_map.get("tail_base")),
    }

    logs.append(f"[KIN] Bodypart mapping: {resolved}")

    def coord_or_nan(bp: Optional[str], coord: str) -> np.ndarray:
        if (not bp) or (bp not in bodyparts):
            return np.full(n, np.nan)
        return get_xy(df, bp, coord)


    # Pick a center bodypart for motion if spine is missing
    center_bp = (
        resolved["spine_upper"]
        or resolved["spine_lower"]
        or resolved["nose"]
        or resolved["left_leg"]
        or resolved["right_leg"]
        or (bodyparts[0] if bodyparts else None)
    )
    if center_bp is None:
        raise RuntimeError("[KIN] No bodyparts found in pose CSV.")

    le_x = coord_or_nan(resolved["left_ear"], "x")
    le_y = coord_or_nan(resolved["left_ear"], "y")
    re_x = coord_or_nan(resolved["right_ear"], "x")
    re_y = coord_or_nan(resolved["right_ear"], "y")
    nose_x = coord_or_nan(resolved["nose"], "x")
    nose_y = coord_or_nan(resolved["nose"], "y")

    neck_x = coord_or_nan(resolved["neck"], "x")
    neck_y = coord_or_nan(resolved["neck"], "y")

    sp_x = coord_or_nan(resolved["spine_upper"], "x")
    sp_y = coord_or_nan(resolved["spine_upper"], "y")
    spL_x = coord_or_nan(resolved["spine_lower"], "x")
    spL_y = coord_or_nan(resolved["spine_lower"], "y")
    center_x = coord_or_nan(center_bp, "x")
    center_y = coord_or_nan(center_bp, "y")

    lShoulder_x = coord_or_nan(resolved["left_shoulder"], "x")
    lShoulder_y = coord_or_nan(resolved["left_shoulder"], "y")
    rShoulder_x = coord_or_nan(resolved["right_shoulder"], "x")
    rShoulder_y = coord_or_nan(resolved["right_shoulder"], "y")

    lLeg_x = coord_or_nan(resolved["left_leg"], "x")
    lLeg_y = coord_or_nan(resolved["left_leg"], "y")
    rLeg_x = coord_or_nan(resolved["right_leg"], "x")
    rLeg_y = coord_or_nan(resolved["right_leg"], "y")

    tailBase_x = coord_or_nan(resolved["tail_base"], "x")
    tailBase_y = coord_or_nan(resolved["tail_base"], "y")
    

    fps = 30.0
    window = 5
    
    # Smooth coordinates to remove jitter
    sp_x_series = pd.Series(center_x).rolling(window, center=True).mean().bfill().ffill()
    sp_y_series = pd.Series(center_y).rolling(window, center=True).mean().bfill().ffill()
    
    sp_x_smooth = sp_x_series.values
    sp_y_smooth = sp_y_series.values

    # Calculate velocity components
    dx = np.diff(sp_x_smooth, prepend=sp_x_smooth[0])
    dy = np.diff(sp_y_smooth, prepend=sp_y_smooth[0])
    
    # Calculate magnitude (Speed)
    speed = np.sqrt(dx**2 + dy**2) * fps
    head_angle_deg = np.full(n, np.nan)
    if resolved["nose"] and resolved["left_ear"] and resolved["right_ear"]:
        ear_mid_x = (le_x + re_x) / 2.0
        ear_mid_y = (le_y + re_y) / 2.0
        head_angle_deg = np.degrees(
            np.arctan2(nose_y - ear_mid_y, nose_x - ear_mid_x)
        )
    elif resolved["nose"]:
        head_angle_deg = np.degrees(
            np.arctan2(nose_y - center_y, nose_x - center_x)
        )
    else:
        # fallback: use motion direction as heading
        dx_h = np.diff(center_x, prepend=center_x[0])
        dy_h = np.diff(center_y, prepend=center_y[0])
        head_angle_deg = np.degrees(np.arctan2(dy_h, dx_h))

    dtheta = np.abs(np.diff(head_angle_deg, prepend=head_angle_deg[0]))

    # ------------------------------------
    # Movement direction (velocity vector)
    # ------------------------------------
    vx = np.diff(center_x, prepend=center_x[0])
    vy = np.diff(center_y, prepend=center_y[0])

    v_norm = np.sqrt(vx**2 + vy**2) + 1e-6
    vx_n = vx / v_norm
    vy_n = vy / v_norm

    # Angle change between consecutive velocity vectors
    dot = vx_n * np.roll(vx_n, 1) + vy_n * np.roll(vy_n, 1)
    dot = np.clip(dot, -1, 1)

    turning_angle_deg = np.degrees(np.arccos(dot))
    turning_angle_deg[0] = 0
    
    # head_spine_angle
    head_spine_angle = np.full(n, np.nan)
    if resolved["nose"] and resolved["spine_upper"] and resolved["spine_lower"]:
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

    out = {
        "frame": np.arange(n),
        "spine_x": center_x,
        "spine_y": center_y,
        "speed_px_s": speed,
        "head_angle_deg": head_angle_deg,
        "turning_rate_deg": dtheta,
        "move_turn_angle_deg": turning_angle_deg,
    }

    if resolved["nose"]:
        out["nose_x"] = nose_x
        out["nose_y"] = nose_y
    if resolved["spine_lower"]:
        out["spine_lower_x"] = spL_x
        out["spine_lower_y"] = spL_y
    if resolved["left_shoulder"]:
        out["left_shoulder_x"] = lShoulder_x
        out["left_shoulder_y"] = lShoulder_y
    if resolved["right_shoulder"]:
        out["right_shoulder_x"] = rShoulder_x
        out["right_shoulder_y"] = rShoulder_y
    if resolved.get("left_leg"):
        out["left_leg_x"] = lLeg_x
        out["left_leg_y"] = lLeg_y
    if resolved.get("right_leg"):
        out["right_leg_x"] = rLeg_x
        out["right_leg_y"] = rLeg_y
    if resolved["tail_base"]:
        out["tailbase_x"] = tailBase_x
        out["tailbase_y"] = tailBase_y
    if resolved["nose"] and resolved["spine_upper"] and resolved["spine_lower"]:
        out["head_spine_angle_deg"] = head_spine_angle

    out = pd.DataFrame(out)

    out.to_csv(kin_cache, index=False)
    logs.append(f"[KIN] Kinematics cached at {kin_cache}")

    return kin_cache
