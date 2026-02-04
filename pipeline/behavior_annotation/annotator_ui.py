# pipeline/behavior_annotation/annotator_ui.py
import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pipeline.behavior_annotation.clip_utils import save_video_clip_ffmpeg

# 너가 이미 만든 함수/모듈들
from pipeline.behavior_annotation.merge_csv import merge_labels_into_ml_features_df
from pipeline.behavior_annotation.mergeRelabelled import overwrite_labels_inplace
from pipeline.ML.randomForest import train_random_forest

BEHAVIOR_LABELS = ["sniffing", "grooming", "rearing", "turning", "moving", "rest", "fast_moving", "other"]
TEMP_CLIP_DIR = os.path.join("temp", "behavior_clips")


def _init_annotator_state():
    """Annotator 페이지에서 필요한 session_state 키를 무조건 만들어 둔다."""
    defaults = {
        "clip_len": 30,
        "clip_stride": 15,
        "clip_start": 0,
        "annotations": [],          # 전체 라벨(clip 단위) 리스트
        "outlier_idx": 0,

        # 경로들 (앱에서 버튼으로 merge/train/retrain 하려면 필요)
        "ann_export_dir": "outputs/annotations",
        "ml_features_csv_path": "",   # 사용자가 넣는 ml_features.csv 경로
        "merged_csv_path": "",        # merge 결과 train_merged.csv 경로
        "rf_out_dir": "outputs/rf",

        # outlier 후보 csv 경로(선택)
        "outlier_csv_path": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _get_video_meta(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, n_frames, w, h


def _load_outlier_candidates(path: str) -> pd.DataFrame | None:
    if not path or not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if "frame" not in df.columns:
            return None
        # clip_start/clip_end 없으면 기본 생성
        if "clip_start" not in df.columns:
            df["clip_start"] = (df["frame"].astype(int) - 15).clip(lower=0)
        if "clip_end" not in df.columns:
            df["clip_end"] = df["clip_start"].astype(int) + 29
        return df
    except Exception:
        return None


def _is_outlier_done(outlier_frame: int, annotations: list[dict]) -> bool:
    return any(int(a["start_frame"]) <= outlier_frame <= int(a["end_frame"]) for a in annotations)


def _ensure_dir(path: str):
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def _save_df_csv(df: pd.DataFrame, folder_or_file: str, filename: str) -> str:
    """folder_or_file 이 폴더면 그 안에 filename으로 저장, 파일이면 그대로 저장"""
    if folder_or_file.lower().endswith(".csv"):
        out_path = folder_or_file
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    else:
        _ensure_dir(folder_or_file)
        out_path = os.path.join(folder_or_file, filename)
    df.to_csv(out_path, index=False)
    return out_path


def _make_clip(video_path: str, start: int, end: int, fps_real: float) -> str:
    _ensure_dir(TEMP_CLIP_DIR)
    clip_path = os.path.abspath(os.path.join(TEMP_CLIP_DIR, f"clip_{start}_{end}.mp4"))
    if (not os.path.exists(clip_path)) or os.path.getsize(clip_path) < 1024:
        save_video_clip_ffmpeg(video_path, start, end, clip_path, fps=fps_real)
    return clip_path


def render_behavior_annotator_page(video_path: str):
    _init_annotator_state()

    st.header("🎬 Behavior Annotator (Clip-based)")

    # video_path 체크
    if not video_path or (not os.path.exists(video_path)):
        st.error("No valid video loaded. 먼저 Home에서 비디오 업로드/분석 후 들어와줘.")
        return

    fps_real, n_frames, _, _ = _get_video_meta(video_path)

    # -----------------------
    # 좌측: 클립 라벨링 UI
    # -----------------------
    with st.expander("⚙️ Clip Settings", expanded=True):
        c1, c2, c3 = st.columns(3)

        clip_len_val = int(st.session_state.get("clip_len", 30))
        stride_val = int(st.session_state.get("clip_stride", 15))

        with c1:
            st.session_state.clip_len = st.number_input(
                "Clip length (frames)", 5, 300, clip_len_val, step=5, key="clip_len_input"
            )
        with c2:
            st.session_state.clip_stride = st.number_input(
                "Stride (frames)", 1, 300, stride_val, step=1, key="clip_stride_input"
            )
        with c3:
            fps_display = st.number_input("FPS (display)", 1, 240, int(round(fps_real)), step=1)

    clip_len = int(st.session_state.clip_len)
    stride = int(st.session_state.clip_stride)

    start = int(st.session_state.clip_start)
    start = max(0, min(start, max(0, n_frames - clip_len)))
    st.session_state.clip_start = start
    end = min(n_frames - 1, start + clip_len - 1)

    clip_path = _make_clip(video_path, start, end, fps_real)

    with open(clip_path, "rb") as f:
        st.video(f.read(), format="video/mp4")

    st.info(f"📍 Frames: {start}-{end} | Time: {start/fps_display:.2f}s-{end/fps_display:.2f}s")

    label = st.selectbox("Label for this clip", BEHAVIOR_LABELS, key="clip_label_select")

    colA, colB, colC, colD = st.columns(4)
    with colA:
        if st.button("✅ Save Label", use_container_width=True):
            st.session_state.annotations.append({
                "start_frame": start,
                "end_frame": end,
                "start_time_s": start / fps_real,
                "end_time_s": end / fps_real,
                "label": label,
            })
            st.toast("Saved!", icon="✅")

    with colB:
        if st.button("⬅️ Prev", use_container_width=True):
            st.session_state.clip_start = max(0, start - stride)
            st.rerun()

    with colC:
        if st.button("Next ➡️", use_container_width=True):
            st.session_state.clip_start = min(n_frames - 1, start + stride)
            st.rerun()

    with colD:
        if st.button("💾 Export Annotations CSV", use_container_width=True):
            if not st.session_state.annotations:
                st.warning("No annotations to save.")
            else:
                ann_df = pd.DataFrame(st.session_state.annotations)
                out_path = _save_df_csv(ann_df, st.session_state.ann_export_dir, "behavior_annotations.csv")
                st.success(f"Saved: {out_path}")

    if st.session_state.annotations:
        with st.expander("📋 Current Annotations Log", expanded=False):
            st.dataframe(pd.DataFrame(st.session_state.annotations), use_container_width=True)

    # -----------------------
    # Sidebar: Outlier Mode (클립 이동만)
    # -----------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Outlier Mode")

    st.session_state.outlier_csv_path = st.sidebar.text_input(
        "Outlier CSV path (optional)",
        value=st.session_state.outlier_csv_path,
        key="outlier_csv_path_input"
    )

    outliers = _load_outlier_candidates(st.session_state.outlier_csv_path)
    if outliers is not None and len(outliers) > 0:
        outliers["done"] = outliers["frame"].apply(lambda f: _is_outlier_done(int(f), st.session_state.annotations))
        n_done = int(outliers["done"].sum())
        st.sidebar.progress(n_done / len(outliers), text=f"Progress: {n_done}/{len(outliers)}")

        idx = int(st.session_state.outlier_idx) % len(outliers)
        cur = outliers.iloc[idx]

        if "pred_label" in outliers.columns:
            st.sidebar.info(f"Model Suggestion: {cur.get('pred_label')} (conf={cur.get('conf', 0):.2f})")

        sc1, sc2 = st.sidebar.columns(2)
        if sc1.button("↩️ Prev Outlier"):
            st.session_state.outlier_idx = (idx - 1) % len(outliers)
            st.session_state.clip_start = int(outliers.iloc[st.session_state.outlier_idx]["clip_start"])
            st.rerun()

        if sc2.button("🎯 Next Outlier"):
            st.session_state.outlier_idx = (idx + 1) % len(outliers)
            st.session_state.clip_start = int(outliers.iloc[st.session_state.outlier_idx]["clip_start"])
            st.rerun()

        if st.sidebar.button("💾 Export Outlier-only CSV"):
            out_frames = set(outliers["frame"].astype(int))
            kept = []
            for a in st.session_state.annotations:
                s = int(a["start_frame"]); e = int(a["end_frame"])
                if any(f in out_frames for f in range(s, e + 1)):
                    kept.append(a)
            if not kept:
                st.sidebar.warning("No outlier annotations found in memory.")
            else:
                out_path = _save_df_csv(pd.DataFrame(kept), st.session_state.ann_export_dir, "behavior_annotations_outliers.csv")
                st.sidebar.success(f"Saved: {out_path}")

    # -----------------------
    # 아래: 모델 파이프라인 탭 (Merge/Train/Overwrite/Retrain)
    # -----------------------
    st.markdown("---")
    st.subheader("🧠 Model Pipeline (Merge → Train / Outlier → Overwrite → Retrain)")

    tab1, tab2 = st.tabs(["1) Initial Train", "2) Outlier Relabel & Retrain"])

    with tab1:
        st.write("처음 라벨링한 annotations로 feature와 merge한 뒤 RandomForest를 학습합니다.")

        st.session_state.ann_export_dir = st.text_input(
            "Annotation export dir",
            value=st.session_state.ann_export_dir,
            key="ann_export_dir_input"
        )

        st.session_state.ml_features_csv_path = st.text_input(
            "ml_features.csv path",
            value=st.session_state.ml_features_csv_path,
            key="ml_features_csv_input"
        )

        st.session_state.rf_out_dir = st.text_input(
            "RF output dir",
            value=st.session_state.rf_out_dir,
            key="rf_out_dir_input"
        )

        boundary_exclude = st.number_input("Boundary exclude frames", 0, 30, 5, step=1, key="boundary_exclude_init")

        cA, cB = st.columns(2)

        with cA:
            if st.button("🔗 Merge (features + annotations)", use_container_width=True):
                if not st.session_state.annotations:
                    st.error("No annotations in memory. Clip 라벨링을 먼저 해줘.")
                elif not st.session_state.ml_features_csv_path or (not os.path.exists(st.session_state.ml_features_csv_path)):
                    st.error("ml_features.csv path is invalid.")
                else:
                    ann_df = pd.DataFrame(st.session_state.annotations)
                    feats_df = pd.read_csv(st.session_state.ml_features_csv_path)

                    merged_df = merge_labels_into_ml_features_df(
                        feats_df,
                        ann_df,
                        mode="onehot",
                        boundary_exclude=int(boundary_exclude),
                        behaviors=BEHAVIOR_LABELS,
                    )

                    _ensure_dir(st.session_state.rf_out_dir)
                    merged_path = os.path.join(st.session_state.rf_out_dir, "train_merged.csv")
                    merged_df.to_csv(merged_path, index=False)
                    st.session_state.merged_csv_path = merged_path

                    st.success(f"Merged saved: {merged_path}")
                    st.dataframe(merged_df.head(50), use_container_width=True)

        with cB:
            if st.button("🟢 Train RandomForest (from merged)", use_container_width=True):
                if not st.session_state.merged_csv_path or (not os.path.exists(st.session_state.merged_csv_path)):
                    st.error("No merged_csv found. Merge를 먼저 실행해줘.")
                else:
                    _ensure_dir(st.session_state.rf_out_dir)
                    train_random_forest(
                        st.session_state.merged_csv_path,
                        st.session_state.rf_out_dir,
                        behaviors=BEHAVIOR_LABELS,
                        use_time_split=True,
                        use_train_mask=True,
                    )
                    st.success("RF trained! outputs/rf 에 모델/리포트 저장됨.")

    with tab2:
        st.write("Outlier에서 다시 라벨링한 clip 결과를 기존 merged csv에 덮어쓴 후 재학습합니다.")

        relabel_csv = st.text_input(
            "Outlier relabel CSV path (behavior_annotations_outliers.csv)",
            value=os.path.join(st.session_state.ann_export_dir, "behavior_annotations_outliers.csv"),
            key="relabel_csv_input"
        )

        overwrite_boundary = st.number_input("Outlier boundary exclude frames", 0, 30, 5, step=1, key="boundary_exclude_outlier")

        c1, c2 = st.columns(2)

        with c1:
            if st.button("🧩 Apply Outlier Relabel (overwrite merged)", use_container_width=True):
                if not st.session_state.merged_csv_path or (not os.path.exists(st.session_state.merged_csv_path)):
                    st.error("No merged_csv found. Initial Train 탭에서 Merge 먼저 해줘.")
                elif not relabel_csv or (not os.path.exists(relabel_csv)):
                    st.error("Outlier relabel CSV path is invalid.")
                else:
                    base_csv = st.session_state.merged_csv_path
                    updated_csv = base_csv.replace(".csv", "_updated.csv")

                    overwrite_labels_inplace(
                        base_csv=base_csv,
                        relabel_csv=relabel_csv,
                        out_csv=updated_csv,
                        behaviors=BEHAVIOR_LABELS,
                        boundary_exclude=int(overwrite_boundary),
                        tie_break="last",
                        update_train_mask=True,
                    )

                    st.session_state.merged_csv_path = updated_csv
                    st.success(f"Updated merged saved: {updated_csv}")

        with c2:
            if st.button("🔁 Re-train RandomForest (updated merged)", use_container_width=True):
                if not st.session_state.merged_csv_path or (not os.path.exists(st.session_state.merged_csv_path)):
                    st.error("No updated merged_csv found. Outlier overwrite 먼저 해줘.")
                else:
                    _ensure_dir(st.session_state.rf_out_dir)
                    train_random_forest(
                        st.session_state.merged_csv_path,
                        st.session_state.rf_out_dir,
                        behaviors=BEHAVIOR_LABELS,
                        use_time_split=True,
                        use_train_mask=True,
                    )
                    st.success("RF re-trained with outlier relabels!")
