import os
import cv2
import pandas as pd
import numpy as np

def render_annotated_video(
    input_video: str,
    pose_csv: str,
    kin_csv: str,
    beh_csv: str | None,
    ml_feat_csv: str,
    logs: list,
    *,
    out_path: str,
    label_map=None,
    roi=None,
    pcutoff: float | None = None,
) -> str:
    """
    Render annotated video (bbox, head direction, speed, behavior).
    Output path is explicitly provided for caching.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ----------------------------
    # Load data
    # ----------------------------
    # DLC CSVs usually have a 3-level multi-index header
    pose = pd.read_csv(pose_csv, header=[0, 1, 2])
    kin = pd.read_csv(kin_csv)
    beh = None
    if beh_csv and os.path.exists(beh_csv):
        beh = pd.read_csv(beh_csv)
    ml_feat = pd.read_csv(ml_feat_csv)

    # ----------------------------
    # Helper: safe keypoint access
    # ----------------------------
    def get_xy(body, coord):
        mask = (
            (pose.columns.get_level_values(1) == body)
            & (pose.columns.get_level_values(2) == coord)
        )
        cols = pose.loc[:, mask]
        if cols.shape[1] == 0:
            return None
        return cols.iloc[:, 0].to_numpy()

    # Keypoints for Bounding Box and Arrows (dynamic)
    available_bodyparts = sorted(set(pose.columns.get_level_values(1)))
    available_bodyparts = [bp for bp in available_bodyparts if isinstance(bp, str) and bp.lower() not in {"bodyparts", "coords", "scorer"}]
    
    keypoints = {}
    for bp in available_bodyparts:
        x = get_xy(bp, "x")
        y = get_xy(bp, "y")
        l = get_xy(bp, "likelihood")
        if x is not None and y is not None:
            keypoints[bp] = {"x": x, "y": y, "likelihood": l}
    
    pose_arrays = []
    for v in keypoints.values():
        pose_arrays.extend([v["x"], v["y"]])

    # ----------------------------
    # Video IO
    # ----------------------------
    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        raise RuntimeError(f"[RENDER] Cannot open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # ----------------------------
    # Frame alignment
    # ----------------------------
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_pose = min((len(a) for a in pose_arrays), default=n_video)
    n_kin = len(kin)
    n_beh = len(beh) if beh is not None else n_kin
    max_frames = min(n_video, n_pose, n_kin, n_beh)

    logs.append(
        f"[RENDER] Frames video={n_video}, pose={n_pose}, "
        f"kin={n_kin}, beh={n_beh} → processing {max_frames} frames"
    )

    # ----------------------------
    # Render loop
    # ----------------------------
    ref_w = roi[2] if roi else w 
    ref_h = roi[3] if roi else h
    scale_x = w / ref_w
    scale_y = h / ref_h
    off_x = roi[0] if roi else 0
    off_y = roi[1] if roi else 0

    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        def transform_coords(raw_x, raw_y):
            # Formula: (Original_Coordinate - Crop_Offset) * Scale_Factor
            new_x = int((raw_x - off_x) * scale_x)
            new_y = int((raw_y - off_y) * scale_y)
            return (new_x, new_y)
        
        # Extract current coordinates
        pts = {}
        def add_pt(key, xarr, yarr, larr):
            if xarr is None or yarr is None:
                return
            x = xarr[i]
            y = yarr[i]
            if larr is not None and pcutoff is not None:
                try:
                    if float(larr[i]) < pcutoff:
                        return
                except Exception:
                    pass
            if np.isnan(x) or np.isnan(y):
                return
            pts[key] = transform_coords(x, y)

        for bp, arrs in keypoints.items():
            add_pt(bp, arrs["x"], arrs["y"], arrs.get("likelihood"))
        raw_pos_x = float(kin.loc[i, "spine_x"]) if "spine_x" in kin.columns else 0.0
        raw_pos_y = float(kin.loc[i, "spine_y"]) if "spine_y" in kin.columns else 0.0
        display_pos = transform_coords(raw_pos_x, raw_pos_y)
        speed = float(kin.loc[i, "speed_px_s"]) if "speed_px_s" in kin.columns else 0.0
        angle = float(kin.loc[i, "head_angle_deg"]) if "head_angle_deg" in kin.columns else 0.0

        # 3. Draw Bounding Box (using already transformed pts)
        if pts:
            xs = [p[0] for p in pts.values()]
            ys = [p[1] for p in pts.values()]
            dynamic_pad = int(15 * scale_x) 
            
            xmin, ymin = max(min(xs) - dynamic_pad, 0), max(min(ys) - dynamic_pad, 0)
            xmax, ymax = min(max(xs) + dynamic_pad, w - 1), min(max(ys) + dynamic_pad, h - 1)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # 4. Draw Keypoints
        for pt in pts.values():
            cv2.circle(frame, pt, 4, (0, 0, 255), -1)
        # 3. Draw Head Direction Arrow (only if nose + both ears are present)
        if "leftEar" in pts and "rightEar" in pts and "nose" in pts:
            emx = int((pts["leftEar"][0] + pts["rightEar"][0]) / 2)
            emy = int((pts["leftEar"][1] + pts["rightEar"][1]) / 2)
            cv2.arrowedLine(frame, (emx, emy), pts["nose"], (255, 0, 0), 2)

        # 4. Behavioral Label Logic
        """cluster_id = int(ml_feat.loc[i, "visual_cluster"])
        if label_map:
            behavior_name = label_map.get(cluster_id, f"Cluster {cluster_id}")
        else:
            behavior_name = f"Cluster {cluster_id}"""

        # 5. Extract Kinematic Data
        pos_x = float(kin.loc[i, "spine_x"]) if "spine_x" in kin.columns else 0.0
        pos_y = float(kin.loc[i, "spine_y"]) if "spine_y" in kin.columns else 0.0
        speed = float(kin.loc[i, "speed_px_s"]) if "speed_px_s" in kin.columns else 0.0
        angle = float(kin.loc[i, "head_angle_deg"]) if "head_angle_deg" in kin.columns else 0.0

        # 6. Text Overlays
        cv2.putText(frame, f"Speed: {speed:.1f}px/s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Head: {angle:.0f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        #cv2.putText(frame, f"Behavior: {behavior_name} (ID: {cluster_id})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Pos: ({display_pos[0]}, {display_pos[1]})", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        out.write(frame)

    # Finalize files
    cap.release()
    out.release()

    logs.append(f"[RENDER] Annotated video saved successfully to {out_path}")
    return out_path
