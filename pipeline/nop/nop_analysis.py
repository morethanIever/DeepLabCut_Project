import os
import numpy as np
import pandas as pd

# -------------------------
# USER-DEFINED PARAMETERS
# -------------------------

FPS = 30

OBJECTS = {
    "left":  {"x": 480, "y": 130},
    "right": {"x": 957, "y": 130},
}

OBJECT_RADIUS = 80                # px
MIN_EXPLORATION_FRAMES = int(0.5 * FPS)   # 0.5 sec


# -------------------------
# Utility
# -------------------------
def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


# -------------------------
# Main NOP analysis
# -------------------------
def run_nop_analysis(kin_csv: str, beh_csv: str, out_dir="outputs/nop"
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    kin = pd.read_csv(kin_csv)
    beh = pd.read_csv(beh_csv)

    df = pd.merge(kin, beh, on="frame")
    n = len(df)
    
    nose_x = df["nose_x"].to_numpy()
    nose_y = df["nose_y"].to_numpy()
    behavior = df["behavior"].to_numpy()

    exploration = {
        "left": np.zeros(n, dtype=bool),
        "right": np.zeros(n, dtype=bool),
    }

    EXCLUDE_BEHAVIORS = {"grooming", "rearing"}
    valid_behavior = ~np.isin(behavior, list(EXCLUDE_BEHAVIORS))

    # -------------------------
    # Distance check
    # -------------------------
    for side, obj in OBJECTS.items():
        d = distance(nose_x, nose_y, obj["x"], obj["y"])
        exploration[side] = ((d < OBJECT_RADIUS) & valid_behavior)

    # -------------------------
    # Extract exploration bouts
    # -------------------------
    events = []

    for side in ["left", "right"]:
        mask = exploration[side]
        count = 0
        start = None

        for i, v in enumerate(mask):
            if v:
                if start is None:
                    start = i
                count += 1
            else:
                if count >= MIN_EXPLORATION_FRAMES:
                    events.append({
                        "object": side,
                        "start_frame": start,
                        "end_frame": i - 1,
                        "duration_s": count / FPS,
                    })
                start = None
                count = 0

        # handle tail
        if count >= MIN_EXPLORATION_FRAMES:
            events.append({
                "object": side,
                "start_frame": start,
                "end_frame": n - 1,
                "duration_s": count / FPS,
            })

    events_df = pd.DataFrame(events)
    events_df.to_csv(f"{out_dir}/nop_events.csv", index=False)

    # -------------------------
    # Summary stats
    # -------------------------
    summary = []

    for side in ["left", "right"]:
        side_events = events_df[events_df["object"] == side]
        total_time = side_events["duration_s"].sum()
        bouts = len(side_events)

        summary.append({
            "object": side,
            "total_exploration_time_s": total_time,
            "exploration_bouts": bouts,
        })

    summary_df = pd.DataFrame(summary)

    # -------------------------
    # NOP Index
    # -------------------------
    t_left = summary_df.loc[summary_df["object"] == "left", "total_exploration_time_s"].values[0]
    t_right = summary_df.loc[summary_df["object"] == "right", "total_exploration_time_s"].values[0]

    nop_index = (t_right - t_left) / (t_right + t_left + 1e-6)

    summary_df["NOP_index"] = nop_index
    summary_df["NOP_ratio"] = t_right / (t_left + 1e-6)
    summary_df["delta_time_s"] = t_right - t_left
    summary_path = os.path.join(out_dir, "nop_summary.csv")
    #summary_df.to_csv(f"{out_dir}/nop_summary.csv", index=False)
    summary_df.to_csv(summary_path, index=False)
    return summary_path