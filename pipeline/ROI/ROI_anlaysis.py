import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import os
import json
from matplotlib.lines import Line2D


def _infer_frame_col(df: pd.DataFrame) -> pd.Series:
    for c in ("frame", "Frame", "frames", "Frames", "Unnamed: 0"):
        if c in df.columns:
            return df[c].astype(int)
    return pd.Series(range(len(df)), name="frame")


def _extract_behavior_series(df: pd.DataFrame) -> pd.Series:
    # Prefer SimBA machine results if present (probabilities + binary labels)
    prob_cols = [c for c in df.columns if str(c).startswith("Probability_")]
    simba_labels = []
    for p in prob_cols:
        label = p.replace("Probability_", "")
        if label in df.columns:
            simba_labels.append(label)
    if simba_labels and prob_cols:
        probs = df[[f"Probability_{l}" for l in simba_labels]].astype(float)
        idx = probs.values.argmax(axis=1)
        labels = [simba_labels[i] for i in idx]
        return pd.Series(labels, index=df.index, name="behavior")

    if "behavior" in df.columns:
        return df["behavior"].astype(str)

    # Prefer targets_inserted binary label columns (Running, Moving, etc.)
    exclude_prefixes = (
        "movement_", "all_bp_", "sum_", "mean_", "low_prob_", "probability_", "prob_"
    )
    exclude_exact = {"frame", "frames", "unnamed: 0", "sum_probabilities", "mean_probabilities"}
    label_cols = []
    for c in df.columns:
        lc = str(c).lower()
        if lc in exclude_exact or lc.startswith(exclude_prefixes):
            continue
        col = df[c]
        if col.dropna().isin([0, 1]).all():
            label_cols.append(c)
    if label_cols:
        labels = []
        sub = df[label_cols]
        for _, row in sub.iterrows():
            ones = [c for c in label_cols if row.get(c, 0) == 1]
            labels.append(ones[0] if ones else "unknown")
        return pd.Series(labels, index=df.index, name="behavior")

    prob_cols = [c for c in df.columns if str(c).lower().startswith(("probability_", "prob_"))]
    if prob_cols:
        probs = df[prob_cols].astype(float)
        idx = probs.values.argmax(axis=1)
        labels = [prob_cols[i] for i in idx]
        labels = [l.replace("Probability_", "").replace("probability_", "").replace("Prob_", "").replace("prob_", "") for l in labels]
        return pd.Series(labels, index=df.index, name="behavior")

    return pd.Series(["unknown"] * len(df), index=df.index, name="behavior")

def run_multi_roi_analysis(
    kin_csv: str,
    roi_list: list,
    fps: int = 30,
    out_dir: str = "outputs/roi",
    beh_csv: str | None = None,
    min_visit_frames: int = 1,
    return_visits: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)
    kin = pd.read_csv(kin_csv)
    kin["frame"] = _infer_frame_col(kin)

    if {"nose_x", "nose_y"}.issubset(kin.columns):
        xy_cols = ("nose_x", "nose_y")
    elif {"spine_x", "spine_y"}.issubset(kin.columns):
        xy_cols = ("spine_x", "spine_y")
    else:
        raise ValueError(
            "kin_csv must contain {'nose_x','nose_y'} or {'spine_x','spine_y'}. "
            f"Got: {set(kin.columns)}"
        )

    if beh_csv and os.path.exists(beh_csv):
        beh = pd.read_csv(beh_csv)
        beh["frame"] = _infer_frame_col(beh)
        beh["behavior"] = _extract_behavior_series(beh)
        df = pd.merge(kin, beh[["frame", "behavior"]], on="frame", how="left")
        df["behavior"] = df["behavior"].fillna("unknown")
    else:
        df = kin.copy()
        df["behavior"] = "unknown"

    nose_x = df[xy_cols[0]].to_numpy()
    nose_y = df[xy_cols[1]].to_numpy()
    speed = df["speed_px_s"].to_numpy() if "speed_px_s" in df.columns else np.full(len(df), np.nan)
    head_dir = df["head_angle_deg"].to_numpy() if "head_angle_deg" in df.columns else np.full(len(df), np.nan)
    turning = df["turning_rate_deg"].to_numpy() if "turning_rate_deg" in df.columns else np.full(len(df), np.nan)
    behavior = df["behavior"].to_numpy()
    frames = df["frame"].to_numpy()

    summary_results = []
    visit_rows = []
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(nose_x, nose_y, alpha=0.4, label="Path")

    for roi in roi_list:
        name = roi.get("name", roi.get("id", "ROI"))
        rtype = roi.get("type", "").lower()

        if rtype == "circle":
            cx = float(roi["cx"])
            cy = float(roi["cy"])
            rad = float(roi.get("radius", 15))

            inside = ((nose_x - cx) ** 2 + (nose_y - cy) ** 2) <= rad ** 2
            ax.add_patch(patches.Circle((cx, cy), rad, linewidth=2, edgecolor="green", facecolor="none"))

        elif rtype == "rect":
            left = float(roi["left"])
            top = float(roi["top"])
            w = float(roi["w"])
            h = float(roi["h"])

            inside = (nose_x >= left) & (nose_x <= left + w) & (nose_y >= top) & (nose_y <= top + h)
            ax.add_patch(patches.Rectangle((left, top), w, h, linewidth=2, edgecolor="red", facecolor="none"))

            cx = left + w / 2.0
            cy = top + h / 2.0

        else:
            continue

        total_frames = int(inside.sum())
        time_s = total_frames / float(fps)

        # Visits = contiguous inside segments
        visits = []
        start = None
        for i, v in enumerate(inside):
            if v and start is None:
                start = i
            elif (not v) and (start is not None):
                if (i - start) >= int(min_visit_frames):
                    visits.append((start, i - 1))
                start = None
        if start is not None:
            if (len(inside) - start) >= int(min_visit_frames):
                visits.append((start, len(inside) - 1))

        for visit_idx, (s, e) in enumerate(visits, start=1):
            beh_slice = behavior[s:e + 1]
            if beh_slice.size > 0:
                vc = pd.Series(beh_slice).value_counts()
                dominant = vc.index[0]
                beh_counts = json.dumps(vc.to_dict(), ensure_ascii=True)
            else:
                dominant = "unknown"
                beh_counts = "{}"

            visit_rows.append({
                "roi_name": name,
                "roi_type": rtype,
                "visit_id": visit_idx,
                "start_frame": int(frames[s]),
                "end_frame": int(frames[e]),
                "enter_frame": int(frames[s]),
                "exit_frame": int(frames[e]),
                "start_time_s": round(float(frames[s]) / fps, 3),
                "end_time_s": round(float(frames[e]) / fps, 3),
                "enter_time_s": round(float(frames[s]) / fps, 3),
                "exit_time_s": round(float(frames[e]) / fps, 3),
                "duration_s": round(float(e - s + 1) / fps, 3),
                "duration_frames": int(e - s + 1),
                "behavior_mode": dominant,
                "behavior_counts": beh_counts,
                "mean_speed_px_s": float(np.nanmean(speed[s:e + 1])) if (e >= s) else np.nan,
                "mean_head_direction_deg": float(np.nanmean(head_dir[s:e + 1])) if (e >= s) else np.nan,
                "mean_turning_rate_deg": float(np.nanmean(turning[s:e + 1])) if (e >= s) else np.nan,
            })

        summary_results.append({
            "ROI Name": name,
            "Type": rtype,
            "Time Spent (s)": round(time_s, 2),
            "Exploring Frames": total_frames,
            "Exploring %": round(100.0 * total_frames / len(df), 2),
            "Visit Count": len(visits),
        })

        ax.text(cx, cy, f"{name}\n{round(time_s,2)}s", ha="center", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.6))

    ax.set_title("Multi-ROI Exploration (inside ROI)")
    ax.invert_yaxis()
    ax.legend(loc="upper right")

    plot_path = os.path.join(out_dir, "multi_roi_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    visits_df = pd.DataFrame(visit_rows)
    visits_path = os.path.join(out_dir, "multi_roi_visits.csv")
    visits_df.to_csv(visits_path, index=False)
    _plot_multi_roi_visits(visits_df, os.path.join(out_dir, "multi_roi_visits_plot.png"))

    if return_visits:
        return pd.DataFrame(summary_results), plot_path, visits_df, visits_path

    return pd.DataFrame(summary_results), plot_path


def _plot_multi_roi_visits(visits_df: pd.DataFrame, out_path: str) -> None:
    if visits_df is None or visits_df.empty:
        return

    required = {"roi_name", "start_time_s", "end_time_s", "behavior_mode"}
    if not required.issubset(set(visits_df.columns)):
        return

    df = visits_df.sort_values("start_time_s").copy()
    roi_names = df["roi_name"].astype(str).unique().tolist()
    roi_palette = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c", "#e67e22"]
    roi_colors = {r: roi_palette[i % len(roi_palette)] for i, r in enumerate(roi_names)}

    behaviors = df["behavior_mode"].astype(str).unique().tolist()
    behav_colors = plt.cm.get_cmap("Set3", max(len(behaviors), 1))
    behav_map = {b: behav_colors(i) for i, b in enumerate(behaviors)}

    fig, axes = plt.subplots(
        4, 1, figsize=(14, 12), sharex=True,
        gridspec_kw={"height_ratios": [1, 2, 2, 2]},
    )

    # Panel 1: Ethogram
    roi_y_map = {r: i + 1 for i, r in enumerate(roi_names)}
    for _, row in df.iterrows():
        axes[0].hlines(
            y=roi_y_map[row["roi_name"]],
            xmin=row["start_time_s"],
            xmax=row["end_time_s"],
            color=behav_map.get(str(row["behavior_mode"]), "gray"),
            linewidth=12,
        )
    axes[0].set_yticks(list(roi_y_map.values()))
    axes[0].set_yticklabels([f"ROI {i+1}" for i in range(len(roi_names))])
    axes[0].set_title("ROI Visits with Kinematics", fontsize=16)

    # Panels 2 & 3: Kinematics with trends
    metrics = [
        ("mean_speed_px_s", "Speed (px/s)", axes[1]),
        ("mean_turning_rate_deg", "Turn Rate (deg/s)", axes[2]),
    ]
    for col_name, label, ax in metrics:
        if col_name not in df.columns:
            ax.set_visible(False)
            continue
        for roi, color in roi_colors.items():
            subset = df[df["roi_name"] == roi]
            ax.scatter(subset["start_time_s"], subset[col_name], color=color, s=60, edgecolors="k", zorder=3)
        trend = df[col_name].rolling(window=3, center=True).mean()
        #ax.plot(df["start_time_s"], trend, color="black", linestyle="--", linewidth=2, label="3-Visit Trend", zorder=2)
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.4)

    # Panel 4: Head direction
    if "mean_head_direction_deg" in df.columns:
        for roi, color in roi_colors.items():
            subset = df[df["roi_name"] == roi]
            axes[3].scatter(
                subset["start_time_s"],
                subset["mean_head_direction_deg"],
                color=color,
                marker="^",
                s=50,
            )
        axes[3].set_ylim(-180, 180)
        axes[3].set_ylabel("Head Dir (deg)")
    else:
        axes[3].set_visible(False)
    axes[3].set_xlabel("Time (s)")

    behav_leg = [Line2D([0], [0], color=behav_map[b], lw=6, label=b) for b in behaviors]
    axes[0].legend(handles=behav_leg, title="Behaviors", loc="center left", bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
