import os
from pathlib import Path
import cv2
import pandas as pd
from typing import Optional, List


def _pick_probability_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        cl = c.lower()
        if cl.startswith("probability_") or cl.startswith("prob_"):
            cols.append(c)
    return cols


def render_simba_overlay(
    input_video: str,
    simba_csv: str,
    *,
    out_path: str,
) -> Optional[str]:
    """
    Render a lightweight overlay video that shows the top SimBA behavior
    (based on probability columns).
    """
    if not os.path.exists(simba_csv):
        return None

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.read_csv(simba_csv)
    prob_cols = _pick_probability_columns(df)
    if not prob_cols:
        return None

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"[SimBA] Cannot open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_pred = len(df)
    max_frames = min(n_video, n_pred)

    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        row = df.loc[i, prob_cols]
        top_col = row.idxmax()
        top_prob = float(row[top_col])
        label = top_col.replace("Probability_", "").replace("prob_", "")

        cv2.putText(
            frame,
            f"SimBA: {label} ({top_prob:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2,
        )
        out.write(frame)

    cap.release()
    out.release()

    return str(Path(out_path).resolve())
