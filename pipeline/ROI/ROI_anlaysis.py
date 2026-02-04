import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import os

def run_multi_roi_analysis(kin_csv: str, roi_list: list, fps: int = 30, out_dir: str = "outputs/roi"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(kin_csv)

    required = {"nose_x", "nose_y"}
    if not required.issubset(df.columns):
        raise ValueError(f"kin_csv must contain {required}. Got: {set(df.columns)}")

    nose_x = df["nose_x"].to_numpy()
    nose_y = df["nose_y"].to_numpy()

    summary_results = []
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

        summary_results.append({
            "ROI Name": name,
            "Type": rtype,
            "Time Spent (s)": round(time_s, 2),
            "Exploring Frames": total_frames,
            "Exploring %": round(100.0 * total_frames / len(df), 2),
        })

        ax.text(cx, cy, f"{name}\n{round(time_s,2)}s", ha="center", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.6))

    ax.set_title("Multi-ROI Exploration (inside ROI)")
    ax.invert_yaxis()
    ax.legend(loc="upper right")

    plot_path = os.path.join(out_dir, "multi_roi_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return pd.DataFrame(summary_results), plot_path
