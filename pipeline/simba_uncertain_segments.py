import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np


def _find_fps(project_dir: str, video_stem: str) -> float:
    info_path = os.path.join(project_dir, "logs", "video_info.csv")
    if not os.path.exists(info_path):
        return 30.0
    try:
        info = pd.read_csv(info_path)
        if "Video" not in info.columns or "fps" not in info.columns:
            return 30.0
        # Try exact match, then fallback without _filtered suffix
        if (info["Video"].astype(str) == video_stem).any():
            return float(info.loc[info["Video"].astype(str) == video_stem, "fps"].iloc[0])
        if video_stem.endswith("_filtered"):
            base = video_stem[: -len("_filtered")]
            if (info["Video"].astype(str) == base).any():
                return float(info.loc[info["Video"].astype(str) == base, "fps"].iloc[0])
    except Exception:
        return 30.0
    return 30.0


def _get_frame_series(df: pd.DataFrame) -> pd.Series:
    for col in ("frame", "Frame", "frames", "Frames", "Unnamed: 0"):
        if col in df.columns:
            return df[col]
    return pd.Series(range(len(df)), name="frame")


def extract_uncertain_segments(
    machine_csv: str,
    *,
    max_prob_low: float = 0.4,
    max_prob_high: float = 0.7,
    gap_thresh: float = 0.15,
    min_duration_sec: float = 0.5,
    output_csv: str | None = None,
) -> str:
    df = pd.read_csv(machine_csv)
    prob_cols = [c for c in df.columns if c.lower().startswith("probability_")]
    if not prob_cols:
        raise RuntimeError("No probability columns found (expected columns starting with 'Probability_').")

    probs = df[prob_cols].to_numpy(dtype=float)
    max_prob = probs.max(axis=1)
    # second max using partial sort
    part = np.partition(probs, -2, axis=1)
    second_max = part[:, -2]
    gap = max_prob - second_max

    uncertain = ((max_prob >= max_prob_low) & (max_prob <= max_prob_high)) | (gap < gap_thresh)

    frames = _get_frame_series(df).astype(int).to_numpy()
    idx = np.where(uncertain)[0]
    if len(idx) == 0:
        out = output_csv or _default_output_path(machine_csv)
        pd.DataFrame(columns=["start_frame", "end_frame", "start_sec", "end_sec", "num_frames"]).to_csv(out, index=False)
        return out

    # group contiguous frames
    groups = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        groups.append((start, prev))
        start = i
        prev = i
    groups.append((start, prev))

    # resolve fps from project dir (assumes machine_results under project_folder/csv)
    project_dir = str(Path(machine_csv).resolve().parents[2])
    video_stem = Path(machine_csv).stem
    fps = _find_fps(project_dir, video_stem)

    rows = []
    for s, e in groups:
        start_frame = int(frames[s])
        end_frame = int(frames[e])
        num_frames = end_frame - start_frame + 1
        duration_sec = num_frames / fps if fps > 0 else 0
        if duration_sec < min_duration_sec:
            continue
        rows.append({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_sec": start_frame / fps if fps > 0 else 0,
            "end_sec": end_frame / fps if fps > 0 else 0,
            "num_frames": num_frames,
            "fps": fps,
        })

    out = output_csv or _default_output_path(machine_csv)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _default_output_path(machine_csv: str) -> str:
    p = Path(machine_csv).resolve()
    out_dir = p.parents[0] / "uncertain_segments"
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{p.stem}_uncertain_segments.csv")


def main():
    ap = argparse.ArgumentParser(description="Extract uncertain behavior segments from SimBA machine_results.")
    ap.add_argument("--machine-csv", required=True, help="Path to SimBA machine_results CSV.")
    ap.add_argument("--max-prob-low", type=float, default=0.4)
    ap.add_argument("--max-prob-high", type=float, default=0.7)
    ap.add_argument("--gap-thresh", type=float, default=0.15)
    ap.add_argument("--min-duration-sec", type=float, default=0.5)
    ap.add_argument("--output-csv", default=None)
    args = ap.parse_args()

    out = extract_uncertain_segments(
        args.machine_csv,
        max_prob_low=args.max_prob_low,
        max_prob_high=args.max_prob_high,
        gap_thresh=args.gap_thresh,
        min_duration_sec=args.min_duration_sec,
        output_csv=args.output_csv,
    )
    print(out)


if __name__ == "__main__":
    main()
