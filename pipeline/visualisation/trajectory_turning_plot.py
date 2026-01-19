import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_trajectory_with_turning_rate(
    kin_csv: str,
    turn_csv: str,
    arena_size=(1280, 720),
    turn_clip=300,     # deg/s (color saturation)
) -> str:
    """
    Plot trajectory colored by turning rate.
    """

    kin = pd.read_csv(kin_csv)
    turn = pd.read_csv(turn_csv)

    # ----------------------------------
    # Align frames
    # ----------------------------------
    df = pd.merge(kin, turn, on="frame", how="inner")
    df = df.sort_values("frame")

    x = df["spine_x"].to_numpy()
    y = df["spine_y"].to_numpy()
    tr = df["turning_rate_deg_s"].to_numpy()

    # ----------------------------------
    # Smooth trajectory (recommended)
    # ----------------------------------
    x = pd.Series(x).rolling(5, center=True).mean().to_numpy()
    y = pd.Series(y).rolling(5, center=True).mean().to_numpy()

    valid = ~np.isnan(x)
    x, y, tr = x[valid], y[valid], tr[valid]

    # clip turning rate for color stability
    tr_clip = np.clip(np.abs(tr), 0, turn_clip)

    # ----------------------------------
    # Plot
    # ----------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))

    sc = ax.scatter(
        x,
        y,
        c=tr_clip,
        cmap="inferno",
        s=6,
        linewidth=0,
    )

    # trajectory line (light)
    ax.plot(x, y, color="gray", alpha=0.3, linewidth=1)

    # start / end
    ax.scatter(x[0], y[0], c="lime", s=40, label="Start", zorder=3)
    ax.scatter(x[-1], y[-1], c="cyan", s=40, label="End", zorder=3)

    # arena
    W, H = arena_size
    ax.add_patch(
        plt.Rectangle((0, 0), W, H, fill=False, linewidth=2)
    )

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("Trajectory colored by Turning Rate")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.grid(alpha=0.3)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Turning rate (deg/s)")

    os.makedirs("outputs/plots", exist_ok=True)
    out_path = "outputs/plots/trajectory_turning_rate.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path
