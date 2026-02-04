# pipeline/behavior_annotation/merge_utils.py
import os
import numpy as np
import pandas as pd

DEFAULT_BEHAVIORS = ["sniffing", "grooming", "rearing", "turning", "moving", "rest", "fast_moving", "other"]

def build_frame_labels_from_clips_df(
    ann_df: pd.DataFrame,
    n_frames: int,
    *,
    behaviors: list[str] | None = None,
    mode: str = "onehot",   # "onehot" or "label"
    default_label: str = "none",
    boundary_exclude: int = 0,
) -> pd.DataFrame:
    if behaviors is None:
        behaviors = DEFAULT_BEHAVIORS

    required = {"start_frame", "end_frame", "label"}
    if not required.issubset(set(ann_df.columns)):
        raise RuntimeError(f"Annotation df must contain columns: {sorted(required)}")

    out = pd.DataFrame({"frame": np.arange(n_frames, dtype=int)})

    # init
    if mode == "label":
        out["behavior_label"] = default_label
    elif mode == "onehot":
        for b in behaviors:
            out[b] = 0
        out["train_mask"] = 0
    else:
        raise ValueError("mode must be 'onehot' or 'label'")

    for _, row in ann_df.iterrows():
        s = int(row["start_frame"])
        e = int(row["end_frame"])
        lab = str(row["label"])

        if e < s:
            s, e = e, s
        s = max(0, s)
        e = min(n_frames - 1, e)
        if s > e:
            continue

        mask = (out["frame"] >= s) & (out["frame"] <= e)

        # boundary exclude => train_mask=1 only for inner region
        if mode == "onehot":
            if boundary_exclude > 0:
                bx = int(boundary_exclude)
                boundary_mask = (
                    ((out["frame"] >= s) & (out["frame"] <= min(n_frames - 1, s + bx))) |
                    ((out["frame"] >= max(0, e - bx)) & (out["frame"] <= e))
                )
                keep_mask = mask & (~boundary_mask)
            else:
                keep_mask = mask

            out.loc[keep_mask, "train_mask"] = 1

        if mode == "label":
            out.loc[mask, "behavior_label"] = lab
        else:
            # overwrite (clip 단위 라벨이니까 reset 후 set)
            out.loc[mask, behaviors] = 0
            if lab in behaviors:
                out.loc[mask, lab] = 1
            else:
                if "other" in behaviors:
                    out.loc[mask, "other"] = 1

    return out


def merge_labels_into_ml_features_df(
    feats_df: pd.DataFrame,
    ann_df: pd.DataFrame,
    *,
    behaviors: list[str] | None = None,
    mode: str = "onehot",
    default_label: str = "none",
    boundary_exclude: int = 0,
) -> pd.DataFrame:
    if behaviors is None:
        behaviors = DEFAULT_BEHAVIORS

    feats = feats_df.copy()
    n_frames = len(feats)

    if "frame" not in feats.columns:
        feats.insert(0, "frame", np.arange(n_frames, dtype=int))
    else:
        feats = feats.sort_values("frame").reset_index(drop=True)

    labels_df = build_frame_labels_from_clips_df(
        ann_df,
        n_frames,
        behaviors=behaviors,
        mode=mode,
        default_label=default_label,
        boundary_exclude=boundary_exclude,
    )

    merged = feats.merge(labels_df, on="frame", how="left")

    # 안전: NaN train_mask -> 0
    if "train_mask" in merged.columns:
        merged["train_mask"] = pd.to_numeric(merged["train_mask"], errors="coerce").fillna(0).astype(int)

    return merged


def merge_labels_into_ml_features_file(
    ml_features_csv: str,
    ann_csv: str,
    out_csv: str,
    *,
    behaviors: list[str] | None = None,
    mode: str = "onehot",
    default_label: str = "none",
    boundary_exclude: int = 0,
) -> str:
    feats = pd.read_csv(ml_features_csv)
    ann = pd.read_csv(ann_csv)

    merged = merge_labels_into_ml_features_df(
        feats, ann,
        behaviors=behaviors,
        mode=mode,
        default_label=default_label,
        boundary_exclude=boundary_exclude,
    )

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return out_csv
