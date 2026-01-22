# pipeline/render_overlay.py
import os
import cv2
import pandas as pd
import numpy as np


def render_annotated_video(
    input_video: str,
    pose_csv: str,
    kin_csv: str,
    beh_csv: str,
    logs: list,
    *,
    out_path: str,
) -> str:
    """
    Render annotated video (bbox, head direction, speed, behavior).
    Output path is explicitly provided for caching.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ----------------------------
    # Load data
    # ----------------------------
    pose = pd.read_csv(pose_csv, header=[0, 1, 2])
    kin = pd.read_csv(kin_csv)
    beh = pd.read_csv(beh_csv)

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

    # Adjust names to your DLC project
    nose_x = get_xy("nose", "x")
    nose_y = get_xy("nose", "y")
    le_x = get_xy("leftEar", "x")
    le_y = get_xy("leftEar", "y")
    re_x = get_xy("rightEar", "x")
    re_y = get_xy("rightEar", "y")
    
    sp_x = get_xy("spineUpper", "x")
    sp_y = get_xy("spineUpper", "y")

    spM_x_arr = get_xy("spineMid", "x")
    spM_y_arr = get_xy("spineMid", "y")
    
    spL_x_arr = get_xy("spineLower", "x")
    spL_y_arr = get_xy("spineLower", "y")

    lShoulder_x_arr = get_xy("leftShoulder", "x")
    lShoulder_y_arr = get_xy("leftShoulder", "y")

    rShoulder_x_arr = get_xy("rightShoulder", "x")
    rShoulder_y_arr = get_xy("rightShoulder", "y")
    """
    lWrist_x_arr = get_xy("leftWrist", "x")
    lWrist_y_arr = get_xy("leftWrist", "y")

    rWrist_x_arr = get_xy("rightWrist", "x")
    rWrist_y_arr = get_xy("rightWrist", "y")
    """

    tailBase_x_arr = get_xy("tailBase", "x")
    tailBase_y_arr = get_xy("tailBase", "y")
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
    # Frame alignment (VERY IMPORTANT)
    # ----------------------------
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_pose = len(nose_x)
    n_kin = len(kin)
    n_beh = len(beh)
    max_frames = min(n_video, n_pose, n_kin, n_beh)

    logs.append(
        f"[RENDER] Frames video={n_video}, pose={n_pose}, "
        f"kin={n_kin}, beh={n_beh} → using {max_frames}"
    )

    # ----------------------------
    # Render loop
    # ----------------------------
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        nx, ny = int(nose_x[i]), int(nose_y[i])
        lx, ly = int(le_x[i]), int(le_y[i])
        rx, ry = int(re_x[i]), int(re_y[i])
        sx, sy = int(sp_x[i]), int(sp_y[i])
        spMx, spMy = int(spM_x_arr[i]), int(spM_y_arr[i])
        spL_x, spL_y = int(spL_x_arr[i]), int(spL_y_arr[i])
        lShoulder_x, lShoulder_y   = int(lShoulder_x_arr[i]), int(lShoulder_y_arr[i])
        rShoulder_x, rShoulder_y   = int(rShoulder_x_arr[i]), int(rShoulder_y_arr[i])
        """
        lWrist_x, lWrist_y   = int(lWrist_x_arr[i]), int(lWrist_y_arr[i])
        rWrist_x, rWrist_y   = int(rWrist_x_arr[i]), int(rWrist_y_arr[i])
        """
        tailBase_x, tailBase_y   = int(tailBase_x_arr[i]), int(tailBase_y_arr[i])
        """
        # draw keypoints
        for (x, y) in [(nx, ny), (lx, ly), (rx, ry), (sx, sy), (spL_x, spL_y), (lShoulder_x, lShoulder_y), (rShoulder_x, rShoulder_y), (lWrist_x, lWrist_y), (rWrist_x, rWrist_y), (tailBase_x, tailBase_y)]:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        # bbox
        
        xs = [nx, lx, rx, sx, spL_x, lShoulder_x, rShoulder_x, lWrist_x, rWrist_x, tailBase_x]
        ys = [ny, ly, ry, sy, spL_y, lShoulder_y, rShoulder_y, lWrist_y, rWrist_y, tailBase_y]
        """
        for (x, y) in [(nx, ny), (lx, ly), (rx, ry), (sx, sy), (spMx, spMy), (spL_x, spL_y), (lShoulder_x, lShoulder_y), (rShoulder_x, rShoulder_y), (tailBase_x, tailBase_y)]:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        xs = [nx, lx, rx, sx, spMx, spL_x, lShoulder_x, rShoulder_x, tailBase_x]
        ys = [ny, ly, ry, sy, spMy, spL_y, lShoulder_y, rShoulder_y, tailBase_y]

        pad = 15
        xmin = max(min(xs) - pad, 0)
        ymin = max(min(ys) - pad, 0)
        xmax = min(max(xs) + pad, w - 1)
        ymax = min(max(ys) + pad, h - 1)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # head direction
        emx = int((lx + rx) / 2)
        emy = int((ly + ry) / 2)
        cv2.arrowedLine(frame, (emx, emy), (nx, ny), (255, 0, 0), 2)
        
        

        # text overlays
        # rodent position (use spine)
        pos_x = float(kin.loc[i, "spine_x"])
        pos_y = float(kin.loc[i, "spine_y"])

        speed = float(kin.loc[i, "speed_px_s"])
        angle = float(kin.loc[i, "head_angle_deg"])
        behavior = str(beh.loc[i, "behavior"])
        confidence = float(beh.loc[i, "confidence"])

        cv2.putText(frame, f"Pos: ({pos_x:.1f}, {pos_y:.1f})", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.putText(frame, f"Speed: {speed:.1f}px/s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Head: {angle:.0f} deg", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(
                    frame,
                    f"Behavior: {behavior} ({confidence:.2f})",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )


        out.write(frame)

    cap.release()
    out.release()

    logs.append(f"[RENDER] Annotated video saved to {out_path}")
    return out_path
