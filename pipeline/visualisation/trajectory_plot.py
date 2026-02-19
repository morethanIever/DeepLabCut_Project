import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_trajectory(kin_csv: str, out_dir: str) -> str:
    df = pd.read_csv(kin_csv)

    x = df["spine_x"].to_numpy()
    y = df["spine_y"].to_numpy()
    speed = df["speed_px_s"].to_numpy()

    # ----------------------------
    # Build line segments for color-mapped trajectory
    # ----------------------------
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    from matplotlib.collections import LineCollection

    lc = LineCollection(
        segments,
        cmap="inferno",
        norm=plt.Normalize(speed.min(), speed.max())
    )
    lc.set_array(speed[:-1])
    lc.set_linewidth(2)

    # ----------------------------
    # Plot
    # ----------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.add_collection(lc)

    # Start / End points
    ax.scatter(x[0], y[0], c="green", s=40, label="Start", zorder=3)
    ax.scatter(x[-1], y[-1], c="red", s=40, label="End", zorder=3)

    ax.invert_yaxis()            # image coordinate system
    ax.set_aspect("equal")
    ax.set_title("Rodent Trajectory (colored by speed)")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    # Colorbar
    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label("Speed (px/s)")

    # ----------------------------
    # Save
    # ----------------------------
    os.makedirs(out_dir, exist_ok=True)
    video_stem = os.path.splitext(os.path.basename(kin_csv))[0]
    out_path = os.path.join(out_dir, f"{video_stem}_trajectory_speed.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path
