import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D


def plot_trajectory_by_behavior(kin_csv: str, beh_csv: str, out_dir:str) -> str:
    # ----------------------------
    # Load & align data
    # ----------------------------
    kin = pd.read_csv(kin_csv)
    beh = pd.read_csv(beh_csv)

    df = pd.merge(kin, beh, on="frame", how="inner")
    df = df.sort_values("frame").reset_index(drop=True)

    # ----------------------------
    # Smooth trajectory (visual only)
    # ----------------------------
    df["spine_x"] = df["spine_x"].rolling(5, center=True).mean()
    df["spine_y"] = df["spine_y"].rolling(5, center=True).mean()
    df = df.dropna().reset_index(drop=True)

    x = df["spine_x"].to_numpy()
    y = df["spine_y"].to_numpy()
    behavior = df["behavior"].to_numpy()

    # ----------------------------
    # Plot setup
    # ----------------------------
    COLOR_MAP = {
        "rest": "blue",
        "move": "green",
        "fast_move": "red",
        "grooming": "yellow",
        "rearing": "orange",
        "turning": "black"
    }

    fig, ax = plt.subplots(figsize=(5, 5))

    # ----------------------------
    # Draw continuous behavior segments
    # ----------------------------
    start = 0
    for i in range(1, len(df)):
        if behavior[i] != behavior[i - 1]:
            ax.plot(
                x[start:i],
                y[start:i],
                color=COLOR_MAP.get(behavior[start], "gray"),
                linewidth=2,
            )
            start = i

    # last segment
    ax.plot(
        x[start:],
        y[start:],
        color=COLOR_MAP.get(behavior[start], "gray"),
        linewidth=2,
    )

    # ----------------------------
    # Start / End markers
    # ----------------------------
    ax.scatter(x[0], y[0], c="black", s=40, label="Start", zorder=3)
    ax.scatter(x[-1], y[-1], c="orange", s=40, label="End", zorder=3)

    # ----------------------------
    # Axes & arena
    # ----------------------------
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("Rodent Trajectory (segmented by behavior)")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.grid(alpha=0.3)

    # Arena boundary (adjust to your video size)
    ax.add_patch(
        plt.Rectangle((0, 0), 1280, 720, fill=False, linewidth=2)
    )

    # ----------------------------
    # Legend
    # ----------------------------
    legend_lines = [
        Line2D([0], [0], color="blue", lw=2, label="rest"),
        Line2D([0], [0], color="green", lw=2, label="move"),
        Line2D([0], [0], color="red", lw=2, label="fast_move"),
        Line2D([0], [0], color="yellow", lw=2, label="grooming"),
        Line2D([0], [0], color="orange", lw=2, label="rearing"),
        Line2D([0], [0], color="black", lw=2, label="turning")
    ]
    ax.legend(handles=legend_lines, loc="upper right")

    # ----------------------------
    # Save
    # ----------------------------
    os.makedirs(out_dir, exist_ok=True)
    video_stem = os.path.splitext(os.path.basename(kin_csv))[0]
    out_path = os.path.join(out_dir, f"{video_stem}_trajectory_behavior.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path
