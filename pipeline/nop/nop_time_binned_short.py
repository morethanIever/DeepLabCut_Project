import os
import numpy as np
import pandas as pd


def c(
    kin_csv: str,
    fps=30,
    out_dir="outputs/nop",
):
    """
    Time-binned NOP for short videos (<30s)
    Uses relative (percentage) bins
    """
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(kin_csv)
    n_frames = len(df)
    duration_s = n_frames / fps

    # relative bins
    bins = [
        (0.0, 0.33, "early"),
        (0.33, 0.66, "middle"),
        (0.66, 1.0, "late"),
    ]

    summary = []

    for start_r, end_r, label in bins:
        f0 = int(start_r * n_frames)
        f1 = int(end_r * n_frames)

        seg = df.iloc[f0:f1]

        if len(seg) == 0:
            continue

        left_time = seg["explore_left"].sum() / fps
        right_time = seg["explore_right"].sum() / fps

        nop = (right_time - left_time) / (right_time + left_time + 1e-6)

        summary.append({
            "bin": label,
            "start_s": f0 / fps,
            "end_s": f1 / fps,
            "left_exploration_s": left_time,
            "right_exploration_s": right_time,
            "NOP_index": nop,
        })

    out_df = pd.DataFrame(summary)
    out_path = os.path.join(out_dir, "nop_time_binned_short.csv")
    out_df.to_csv(out_path, index=False)

    return out_path
