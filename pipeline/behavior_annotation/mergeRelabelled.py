#!/usr/bin/env python3
# pipeline/behavior_annotation/mergeRelabelled.py

import os
import argparse
import numpy as np
import pandas as pd
from typing import List

DEFAULT_BEHAVIORS = ["sniffing", "grooming", "rearing", "turning", "moving", "rest", "fast_moving", "other"]

def load_clips(ann_csv: str) -> pd.DataFrame:
    ann = pd.read_csv(ann_csv)
    required = {"start_frame", "end_frame", "label"}
    if not required.issubset(set(ann.columns)):
        raise RuntimeError(f"Annotation CSV must have columns: {sorted(required)}")
    ann = ann.copy()
    ann["start_frame"] = ann["start_frame"].astype(int)
    ann["end_frame"] = ann["end_frame"].astype(int)
    ann["label"] = ann["label"].astype(str)
    return ann

def _normalize_label(label: str, behaviors: List[str]) -> str:
    if label in behaviors:
        return label
    return "other" if "other" in behaviors else behaviors[0]

def overwrite_labels_inplace(
    base_csv: str,
    relabel_csv: str,
    out_csv: str,
    *,
    behaviors: List[str],
    boundary_exclude: int = 3,
    tie_break: str = "last",  # outlier relabel은 "마지막 것이 정답"이 일반적으로 자연스러움
    update_train_mask: bool = True,
):
    """
    base_csv: 이미 feature + onehot + train_mask (혹은 behavior_label) 있는 원본(merged) 파일
    relabel_csv: outlier clip relabel 결과 (start_frame,end_frame,label ...)
    out_csv: 저장 경로 (base_csv를 덮어쓰고 싶으면 out_csv=base_csv로 주면 됨)

    동작:
      - relabel clip에 해당하는 프레임 구간만 onehot / behavior_label 덮어쓰기
      - boundary_exclude만큼 clip 경계 프레임은 train_mask=0 (update_train_mask=True일 때)
      - 나머지 프레임은 base_csv 값 유지
    """
    df = pd.read_csv(base_csv)
    if "frame" not in df.columns:
        raise RuntimeError("base_csv must have a 'frame' column")

    # onehot 컬럼이 없으면 만들어 둠(0으로)
    for b in behaviors:
        if b not in df.columns:
            df[b] = 0

    if "behavior_label" not in df.columns:
        df["behavior_label"] = "none"

    if "train_mask" not in df.columns:
        df["train_mask"] = 0

    clips = load_clips(relabel_csv)

    # 빠른 접근을 위해 frame -> row index 맵 만들기
    # (frame이 연속 0..N-1이면 그냥 df.index로 충분하지만 안전하게 매핑)
    frame_to_idx = {int(f): i for i, f in enumerate(df["frame"].astype(int).tolist())}

    # --- 덮어쓰기 전략 ---
    # clip들이 겹치거나 중복될 수 있음.
    # tie_break="last": CSV에서 나중에 등장한 clip이 최종 라벨을 덮어씀 (가장 직관적)
    # tie_break="vote": 프레임마다 다수결로 결정 (원하면 추가 구현 가능)
    if tie_break not in ("last",):
        raise RuntimeError("For overwrite mode, tie_break supports only: 'last' (recommended).")

    # 1) 먼저: relabel 구간 프레임들은 일단 모든 onehot=0으로 초기화 (해당 구간만)
    affected_frames = set()
    for _, row in clips.iterrows():
        s = int(row["start_frame"]); e = int(row["end_frame"])
        if e < s: s, e = e, s
        for f in range(s, e + 1):
            if f in frame_to_idx:
                affected_frames.add(f)

    # affected 프레임들만 onehot 초기화 + behavior_label none
    for f in affected_frames:
        i = frame_to_idx[f]
        for b in behaviors:
            df.at[i, b] = 0
        df.at[i, "behavior_label"] = "none"

        # train_mask는 update_train_mask=True면 아래에서 다시 세팅.
        # 여기서는 건드리지 않음(기본 유지)

    # 2) 그 다음: CSV 순서대로 clip을 적용(나중 clip이 덮어씀)
    # train_mask는 clip 내부는 1, 경계 제외는 0로 (옵션)
    for _, row in clips.iterrows():
        s = int(row["start_frame"]); e = int(row["end_frame"])
        if e < s: s, e = e, s
        lab = _normalize_label(str(row["label"]), behaviors)

        # boundary 제외 구간 계산
        inner_s = s + boundary_exclude
        inner_e = e - boundary_exclude
        # (경계 제외 때문에 inner가 깨질 수 있음)
        has_inner = inner_s <= inner_e

        # 전체 구간 프레임: 라벨은 모두 덮어쓰되,
        # train_mask는 inner만 1, boundary는 0 (옵션)
        for f in range(s, e + 1):
            if f not in frame_to_idx:
                continue
            i = frame_to_idx[f]

            # onehot 덮어쓰기
            for b in behaviors:
                df.at[i, b] = 1 if b == lab else 0
            df.at[i, "behavior_label"] = lab

            if update_train_mask:
                if boundary_exclude <= 0:
                    df.at[i, "train_mask"] = 1
                else:
                    if has_inner and (inner_s <= f <= inner_e):
                        df.at[i, "train_mask"] = 1
                    else:
                        df.at[i, "train_mask"] = 0

    # 3) 저장
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)

    # 간단 sanity check 출력
    # ---- robust sanity check ----
    lbl = df[behaviors].apply(pd.to_numeric, errors="coerce").fillna(0)
    df["train_mask"] = pd.to_numeric(df["train_mask"], errors="coerce").fillna(0).astype(int)

    labeled = (lbl.sum(axis=1) > 0).sum()
    usable = ((lbl.sum(axis=1) > 0) & (df["train_mask"] == 1)).sum()

    print(f"[OVERWRITE] saved -> {out_csv}")
    print(f"[OVERWRITE] labeled frames total: {labeled}")
    print(f"[OVERWRITE] trainable frames total (train_mask==1): {usable}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_csv", required=True, help="original merged csv (features + labels)")
    p.add_argument("--relabel_csv", required=True, help="outlier relabel clip csv (start_frame,end_frame,label)")
    p.add_argument("--out_csv", required=True, help="output csv (can be same as base_csv to overwrite file)")
    p.add_argument("--boundary_exclude", type=int, default=3)
    p.add_argument("--tie_break", choices=["last"], default="last")
    p.add_argument("--no_train_mask_update", action="store_true", help="do not change train_mask; only update onehot/behavior_label")
    p.add_argument("--behaviors", nargs="+", default=DEFAULT_BEHAVIORS)
    args = p.parse_args()

    overwrite_labels_inplace(
        base_csv=args.base_csv,
        relabel_csv=args.relabel_csv,
        out_csv=args.out_csv,
        behaviors=args.behaviors,
        boundary_exclude=args.boundary_exclude,
        tie_break=args.tie_break,
        update_train_mask=(not args.no_train_mask_update),
    )

if __name__ == "__main__":
    main()
