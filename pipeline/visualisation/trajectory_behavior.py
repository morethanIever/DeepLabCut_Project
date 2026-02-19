import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from matplotlib.collections import LineCollection


def _infer_frame_col(df: pd.DataFrame) -> pd.Series:
    for c in ("frame", "Frame", "frames", "Frames", "Unnamed: 0"):
        if c in df.columns:
            return df[c].astype(int)
    return pd.Series(range(len(df)), name="frame")


def _extract_behavior_series(df: pd.DataFrame) -> pd.Series:
    # 1) Explicit behavior column (legacy behavior csv)
    if "behavior" in df.columns:
        return df["behavior"].astype(str)

    # 2) SimBA probabilities (Probability_x)
    prob_cols = [c for c in df.columns if c.lower().startswith("probability_") or c.lower().startswith("prob_")]
    if prob_cols:
        probs = df[prob_cols].astype(float)
        idx = probs.values.argmax(axis=1)
        labels = [prob_cols[i] for i in idx]
        labels = [l.replace("Probability_", "").replace("probability_", "").replace("Prob_", "").replace("prob_", "") for l in labels]
        return pd.Series(labels, index=df.index, name="behavior")

    # 3) SimBA binary columns (e.g., rearing, turning, etc.)
    candidate_labels = []
    for c in df.columns:
        lc = str(c).lower()
        if lc in {"frame", "frames", "unnamed: 0"}:
            continue
        if lc.startswith("movement_") or lc.startswith("all_bp_") or lc.startswith("sum_") or lc.startswith("mean_"):
            continue
        # likely class label columns
        if df[c].dropna().isin([0, 1]).all():
            candidate_labels.append(c)
    if candidate_labels:
        labels = []
        sub = df[candidate_labels]
        for _, row in sub.iterrows():
            ones = [c for c in candidate_labels if row.get(c, 0) == 1]
            labels.append(ones[0] if ones else "unknown")
        return pd.Series(labels, index=df.index, name="behavior")

    # Fallback
    return pd.Series(["unknown"] * len(df), index=df.index, name="behavior")


def plot_trajectory_by_behavior(kin_csv: str, beh_csv: str, out_dir: str) -> str:
    # ----------------------------
    # Load & align data
    # ----------------------------
    kin = pd.read_csv(kin_csv)
    beh = pd.read_csv(beh_csv)
    beh["frame"] = _infer_frame_col(beh)
    beh["behavior"] = _extract_behavior_series(beh)

    df = pd.merge(kin, beh, on="frame", how="inner")
    df = df.sort_values("frame").reset_index(drop=True)

    # ----------------------------
    # Smooth trajectory (visual only)
    # ----------------------------
    df["spine_x"] = df["spine_x"].rolling(5, center=True, min_periods=1).mean()
    df["spine_y"] = df["spine_y"].rolling(5, center=True, min_periods=1).mean()

    x = df["spine_x"].to_numpy()
    y = df["spine_y"].to_numpy()
    behavior = df["behavior"].to_numpy()

    # ----------------------------
    # Plot setup
    # ----------------------------
    # Build a color map dynamically from observed behaviors
    unique_behaviors = sorted(set(behavior))
    base_colors = cm.get_cmap("tab20", max(1, len(unique_behaviors)))
    COLOR_MAP = {
        b: base_colors(i) for i, b in enumerate(unique_behaviors)
    }

    fig, ax = plt.subplots(figsize=(5, 5))

    # ----------------------------
    # Draw continuous per-frame segments (no gaps between behavior changes)
    # ----------------------------
    if len(x) > 1:
        points = np.column_stack([x, y])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        seg_colors = [COLOR_MAP.get(behavior[i], "gray") for i in range(len(behavior) - 1)]
        lc = LineCollection(segments, colors=seg_colors, linewidths=2)
        ax.add_collection(lc)

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
        Line2D([0], [0], color=COLOR_MAP.get(b, "gray"), lw=2, label=b)
        for b in unique_behaviors
    ]
    ax.legend(
        handles=legend_lines,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    # ----------------------------
    # Save
    # ----------------------------
    os.makedirs(out_dir, exist_ok=True)
    video_stem = os.path.splitext(os.path.basename(kin_csv))[0]
    out_path = os.path.join(out_dir, f"{video_stem}_trajectory_behavior.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path
