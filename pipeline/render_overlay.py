import os
import cv2
import pandas as pd
import numpy as np

def render_annotated_video(
    input_video: str,
    pose_csv: str,
    kin_csv: str,
    beh_csv: str,
    ml_feat_csv: str,
    logs: list,
    *,
    out_path: str,
    label_map=None
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
            raise RuntimeError(
                f"[RENDER] Bodypart '{body}' with coord '{coord}' not found.\n"
                f"Available bodyparts: {sorted(set(pose.columns.get_level_values(1)))}"
            )
        return cols.iloc[:, 0].to_numpy()

    # Keypoints for Bounding Box and Arrows
    nose_x = get_xy("nose", "x")
    nose_y = get_xy("nose", "y")
    le_x = get_xy("leftEar", "x")
    le_y = get_xy("leftEar", "y")
    re_x = get_xy("rightEar", "x")
    re_y = get_xy("rightEar", "y")
    sp_x = get_xy("spineUpper", "x")
    sp_y = get_xy("spineUpper", "y")
    spMx = get_xy("spineMid", "x")
    spMy = get_xy("spineMid", "y")
    spLx = get_xy("spineLower", "x")
    spLy = get_xy("spineLower", "y")
    ls_x = get_xy("leftShoulder", "x")
    ls_y = get_xy("leftShoulder", "y")
    rs_x = get_xy("rightShoulder", "x")
    rs_y = get_xy("rightShoulder", "y")
    tb_x = get_xy("tailBase", "x")
    tb_y = get_xy("tailBase", "y")

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
    n_pose = len(nose_x)
    n_kin = len(kin)
    n_beh = len(beh)
    max_frames = min(n_video, n_pose, n_kin, n_beh)

    logs.append(
        f"[RENDER] Frames video={n_video}, pose={n_pose}, "
        f"kin={n_kin}, beh={n_beh} → processing {max_frames} frames"
    )

    # ----------------------------
    # Render loop
    # ----------------------------
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Extract current coordinates
        pts = {
            "nose": (int(nose_x[i]), int(nose_y[i])),
            "le": (int(le_x[i]), int(le_y[i])),
            "re": (int(re_x[i]), int(re_y[i])),
            "spU": (int(sp_x[i]), int(sp_y[i])),
            "spM": (int(spMx[i]), int(spMy[i])),
            "spL": (int(spLx[i]), int(spLy[i])),
            "ls": (int(ls_x[i]), int(ls_y[i])),
            "rs": (int(rs_x[i]), int(rs_y[i])),
            "tb": (int(tb_x[i]), int(tb_y[i]))
        }

        # 1. Draw Keypoints
        for pt in pts.values():
            cv2.circle(frame, pt, 4, (0, 0, 255), -1)

        # 2. Draw Bounding Box
        xs = [p[0] for p in pts.values()]
        ys = [p[1] for p in pts.values()]
        pad = 15
        xmin, ymin = max(min(xs) - pad, 0), max(min(ys) - pad, 0)
        xmax, ymax = min(max(xs) + pad, w - 1), min(max(ys) + pad, h - 1)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # 3. Draw Head Direction Arrow
        emx = int((pts["le"][0] + pts["re"][0]) / 2)
        emy = int((pts["le"][1] + pts["re"][1]) / 2)
        cv2.arrowedLine(frame, (emx, emy), pts["nose"], (255, 0, 0), 2)

        # 4. Behavioral Label Logic
        cluster_id = int(ml_feat.loc[i, "visual_cluster"])
        if label_map:
            behavior_name = label_map.get(cluster_id, f"Cluster {cluster_id}")
        else:
            behavior_name = f"Cluster {cluster_id}"

        # 5. Extract Kinematic Data
        pos_x, pos_y = float(kin.loc[i, "spine_x"]), float(kin.loc[i, "spine_y"])
        speed = float(kin.loc[i, "speed_px_s"])
        angle = float(kin.loc[i, "head_angle_deg"])

        # 6. Text Overlays
        cv2.putText(frame, f"Speed: {speed:.1f}px/s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Head: {angle:.0f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Behavior: {behavior_name} (ID: {cluster_id})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Pos: ({pos_x:.1f}, {pos_y:.1f})", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        out.write(frame)

    # Finalize files
    cap.release()
    out.release()

    logs.append(f"[RENDER] Annotated video saved successfully to {out_path}")
    return out_path