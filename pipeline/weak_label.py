import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _parse_labels(raw: Optional[List[str]]) -> List[str]:
    if not raw:
        return []
    if len(raw) == 1:
        return [x.strip() for x in raw[0].split(",") if x.strip()]
    labels: List[str] = []
    for item in raw:
        labels.extend([x.strip() for x in item.split(",") if x.strip()])
    return labels


def _get_frame_series(df: pd.DataFrame, frame_col: str = "frame") -> pd.Series:
    if frame_col in df.columns:
        return df[frame_col]
    for col in ("Frame", "frames", "Frames", "Unnamed: 0"):
        if col in df.columns:
            return df[col]
    return pd.Series(range(len(df)), name=frame_col)


def _find_fps(machine_csv: str, video_stem: str) -> float:
    # Expect .../project_folder/csv/machine_results/<file>.csv
    try:
        project_dir = str(Path(machine_csv).resolve().parents[2])
    except Exception:
        return 0.0
    info_path = os.path.join(project_dir, "logs", "video_info.csv")
    if not os.path.exists(info_path):
        return 0.0
    try:
        info = pd.read_csv(info_path)
        if "Video" not in info.columns or "fps" not in info.columns:
            return 0.0
        if (info["Video"].astype(str) == video_stem).any():
            return float(info.loc[info["Video"].astype(str) == video_stem, "fps"].iloc[0])
        if video_stem.endswith("_filtered"):
            base = video_stem[: -len("_filtered")]
            if (info["Video"].astype(str) == base).any():
                return float(info.loc[info["Video"].astype(str) == base, "fps"].iloc[0])
    except Exception:
        return 0.0
    return 0.0


def _enforce_min_bout(mask: np.ndarray, min_len: int) -> np.ndarray:
    if min_len <= 1:
        return mask
    out = np.zeros_like(mask, dtype=bool)
    count = 0
    for i, v in enumerate(mask):
        if v:
            count += 1
        else:
            count = 0
        if count >= min_len:
            out[i - min_len + 1 : i + 1] = True
    return out


def _find_prob_column(df: pd.DataFrame, label: str) -> Optional[str]:
    label_l = label.lower()
    cols = list(df.columns)
    # direct candidates
    direct = [
        f"probability_{label}",
        f"prob_{label}",
        label,
    ]
    for c in direct:
        if c in cols:
            return c
    # case-insensitive match
    for c in cols:
        if c.lower() == label_l:
            return c
    # probability prefix match
    for c in cols:
        cl = c.lower()
        if cl.startswith("probability_") or cl.startswith("prob_"):
            if cl.endswith(label_l):
                return c
    return None


def build_pseudo_annotations(
    *,
    machine_csv: str,
    labels: List[str],
    threshold: float,
    min_bout: int,
    out_csv: str,
    frame_col: str = "frame",
) -> str:
    if not os.path.exists(machine_csv):
        raise FileNotFoundError(f"machine_csv not found: {machine_csv}")
    if not labels:
        raise ValueError("No labels provided.")

    df = pd.read_csv(machine_csv)
    frames = _get_frame_series(df, frame_col=frame_col).astype(int).to_numpy()

    rows: List[Dict[str, int | str]] = []
    for lab in labels:
        col = _find_prob_column(df, lab)
        if not col:
            raise ValueError(f"Probability column not found for label '{lab}' in {machine_csv}")
        prob = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
        mask = prob >= float(threshold)
        mask = _enforce_min_bout(mask, int(min_bout))

        in_bout = False
        start_idx = 0
        for i, v in enumerate(mask):
            if v and not in_bout:
                in_bout = True
                start_idx = i
            elif (not v) and in_bout:
                end_idx = i - 1
                rows.append({
                    "start_frame": int(frames[start_idx]),
                    "end_frame": int(frames[end_idx]),
                    "label": str(lab),
                })
                in_bout = False
        if in_bout:
            rows.append({
                "start_frame": int(frames[start_idx]),
                "end_frame": int(frames[len(mask) - 1]),
                "label": str(lab),
            })

    ann_df = pd.DataFrame(rows, columns=["start_frame", "end_frame", "label"])
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    ann_df.to_csv(out_csv, index=False)
    return out_csv


def build_windowed_annotations(
    *,
    machine_csv: str,
    labels: List[str],
    window_len: int,
    stride: int,
    out_csv: str,
    threshold: float | None = None,
    none_label: str = "none",
    frame_col: str = "frame",
) -> str:
    if not os.path.exists(machine_csv):
        raise FileNotFoundError(f"machine_csv not found: {machine_csv}")
    if not labels:
        raise ValueError("No labels provided.")
    window_len = max(1, int(window_len))
    stride = max(1, int(stride))

    df = pd.read_csv(machine_csv)
    frames = _get_frame_series(df, frame_col=frame_col).astype(int).to_numpy()
    n = len(df)
    video_stem = os.path.splitext(os.path.basename(machine_csv))[0]
    fps = _find_fps(machine_csv, video_stem)

    prob_arrays = []
    for lab in labels:
        col = _find_prob_column(df, lab)
        if not col:
            raise ValueError(f"Probability column not found for label '{lab}' in {machine_csv}")
        prob = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
        prob_arrays.append(prob)

    rows: List[Dict[str, int | str]] = []
    for start_idx in range(0, n, stride):
        end_idx = min(n - 1, start_idx + window_len - 1)
        if end_idx < start_idx:
            break
        # mean probability per label in this window
        means = [float(p[start_idx : end_idx + 1].mean()) for p in prob_arrays]
        best_i = int(np.argmax(means)) if means else 0
        best_prob = means[best_i] if means else 0.0
        label = labels[best_i] if means else none_label
        if threshold is not None and best_prob < float(threshold):
            label = none_label

        rows.append({
            "start_frame": int(frames[start_idx]),
            "end_frame": int(frames[end_idx]),
            "start_time_s": float(frames[start_idx]) / fps if fps > 0 else 0.0,
            "end_time_s": float(frames[end_idx]) / fps if fps > 0 else 0.0,
            "label": str(label),
        })

        if end_idx == n - 1:
            break

    ann_df = pd.DataFrame(rows, columns=["start_frame", "end_frame", "start_time_s", "end_time_s", "label"])
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    ann_df.to_csv(out_csv, index=False)
    return out_csv


def build_pseudo_targets(
    *,
    machine_csv: str,
    ml_features_csv: Optional[str],
    labels: List[str],
    threshold: float,
    min_bout: int,
    out_dir: str,
    frame_col: str = "frame",
) -> str:
    if not os.path.exists(machine_csv):
        raise FileNotFoundError(f"machine_csv not found: {machine_csv}")

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(machine_csv)
    if frame_col not in df.columns:
        df.insert(0, frame_col, np.arange(len(df), dtype=int))

    label_cols: Dict[str, str] = {}
    for lab in labels:
        col = _find_prob_column(df, lab)
        if not col:
            raise ValueError(f"Probability column not found for label '{lab}' in {machine_csv}")
        label_cols[lab] = col

    out_labels = pd.DataFrame({frame_col: df[frame_col].astype(int)})
    for lab, col in label_cols.items():
        prob = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
        mask = prob >= threshold
        mask = _enforce_min_bout(mask, min_bout)
        out_labels[lab] = mask.astype(int)

    if ml_features_csv:
        if not os.path.exists(ml_features_csv):
            raise FileNotFoundError(f"ml_features_csv not found: {ml_features_csv}")
        feats = pd.read_csv(ml_features_csv)
        if frame_col not in feats.columns:
            feats.insert(0, frame_col, np.arange(len(feats), dtype=int))
        merged = feats.merge(out_labels, on=frame_col, how="left")
        merged = merged.fillna(0)
        out_df = merged
    else:
        out_df = out_labels

    stem = os.path.splitext(os.path.basename(machine_csv))[0]
    out_path = os.path.join(out_dir, f"{stem}.csv")
    out_df.to_csv(out_path, index=False)
    return out_path


def _default_out_dir(machine_csv: str) -> str:
    # expect .../project_folder/csv/machine_results/<file>.csv
    root = os.path.abspath(machine_csv)
    parts = root.split(os.sep)
    if "csv" in parts:
        csv_idx = len(parts) - 1 - parts[::-1].index("csv")
        project_dir = os.sep.join(parts[:csv_idx])
        return os.path.join(project_dir, "csv", "targets_inserted_pseudo")
    return os.path.join(os.path.dirname(root), "targets_inserted_pseudo")


def main() -> int:
    p = argparse.ArgumentParser(description="Create pseudo-label targets_inserted from SimBA machine results")
    p.add_argument("--machine-csv", required=True, help="Path to SimBA machine_results CSV")
    p.add_argument("--ml-features", default="", help="Path to ml_features CSV to merge (optional)")
    p.add_argument("--labels", nargs="*", default=None, help="Labels (space- or comma-separated)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-bout", type=int, default=5)
    p.add_argument("--out-dir", default="", help="Output directory (optional)")
    args = p.parse_args()

    labels = _parse_labels(args.labels)
    if not labels:
        raise ValueError("No labels provided. Use --labels label1,label2,...")

    out_dir = args.out_dir.strip() or _default_out_dir(args.machine_csv)
    out_path = build_pseudo_targets(
        machine_csv=args.machine_csv,
        ml_features_csv=args.ml_features or None,
        labels=labels,
        threshold=args.threshold,
        min_bout=args.min_bout,
        out_dir=out_dir,
    )
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
