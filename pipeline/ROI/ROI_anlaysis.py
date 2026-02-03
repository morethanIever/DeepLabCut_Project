import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import os

def run_multi_roi_analysis(
    kin_csv: str,
    roi_list: list,
    radius: float,
    fps: int = 30,
    out_dir: str = "outputs/roi",
):
    """
    kin_csv: kinematics csv path (must contain nose_x, nose_y)
    roi_list: from ROI editor, supports circle & rect
    radius: exploration distance threshold (px) used for ALL ROIs
    """
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(kin_csv)

    # ✅ 프로젝트에 맞게 컬럼 체크
    required = {"nose_x", "nose_y"}
    if not required.issubset(df.columns):
        raise ValueError(f"kin_csv must contain {required}. Got: {set(df.columns)}")

    nose_x = df["nose_x"].to_numpy()
    nose_y = df["nose_y"].to_numpy()

    summary_results = []

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(nose_x, nose_y, alpha=0.4, label="Path")

    for roi in roi_list:
        name = roi.get("name", "ROI")
        cx = float(roi["cx"])
        cy = float(roi["cy"])

        # ✅ 거리 기반 탐색 계산 (circle/rect 모두 중심 기준)
        dist = np.sqrt((nose_x - cx) ** 2 + (nose_y - cy) ** 2)
        exploring = dist < float(radius)
        total_frames = int(exploring.sum())
        time_s = total_frames / float(fps)

        summary_results.append({
            "ROI Name": name,
            "Type": roi.get("type"),
            "CX": round(cx, 2),
            "CY": round(cy, 2),
            "Time Spent (s)": round(time_s, 2),
            "Exploring Frames": total_frames,
            "Avg Distance (px)": round(float(dist.mean()), 2),
        })

        # ✅ 그리기: 원/사각형 모양 그대로 + 탐색 반경(원)
        if roi.get("type") == "circle":
            rad0 = float(roi.get("radius") or 15)
            shape = patches.Circle((cx, cy), rad0, linewidth=2, edgecolor="green", facecolor="none")
            ax.add_patch(shape)
        elif roi.get("type") == "rect":
            left = float(roi.get("left", cx - 15))
            top = float(roi.get("top", cy - 15))
            w = float(roi.get("w", 30))
            h = float(roi.get("h", 30))
            shape = patches.Rectangle((left, top), w, h, linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(shape)

        # 탐색 반경 표시
        ax.add_patch(patches.Circle((cx, cy), float(radius), color="blue", alpha=0.08))

        ax.text(cx, cy - float(radius) - 5,
                f"{name}\n{round(time_s,2)}s",
                ha="center", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.6))

    ax.set_title(f"Multi-ROI Exploration (radius={radius}px)")
    ax.invert_yaxis()
    ax.legend(loc="upper right")

    plot_path = os.path.join(out_dir, "multi_roi_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return pd.DataFrame(summary_results), plot_path
