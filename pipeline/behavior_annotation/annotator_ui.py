# pipeline/behavior_annotation/annotator_ui.py
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from pipeline.behavior_annotation.clip_utils import save_video_clip_ffmpeg

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

@st.cache_data(show_spinner=False)
def _read_frame_cached(video_path: str, frame_idx: int):
    # cache: video_path+frame_idx -> PIL image
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_idx}")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

def _init_ann_df(n_frames: int, behaviors: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({"frame": np.arange(n_frames, dtype=int)})
    for b in behaviors:
        df[b] = 0
    return df

def _ensure_state(video_path: str, behaviors: list[str]):
    # Initialize annotation table per video+behavior set
    if "ann_video_path" not in st.session_state:
        st.session_state.ann_video_path = None
    if "ann_behaviors" not in st.session_state:
        st.session_state.ann_behaviors = []
    if "ann_df" not in st.session_state:
        st.session_state.ann_df = None
    if "ann_frame" not in st.session_state:
        st.session_state.ann_frame = 0
    if "ann_jump" not in st.session_state:
        st.session_state.ann_jump = 1
    if "ann_message" not in st.session_state:
        st.session_state.ann_message = ""

    # reset condition: new video OR behaviors changed
    if st.session_state.ann_video_path != video_path or st.session_state.ann_behaviors != behaviors:
        fps, n_frames, _, _ = _get_video_meta(video_path)
        st.session_state.ann_video_path = video_path
        st.session_state.ann_behaviors = behaviors
        st.session_state.ann_df = _init_ann_df(n_frames, behaviors)
        st.session_state.ann_frame = 0
        st.session_state.ann_jump = 1
        st.session_state.ann_message = f"Initialized annotator. FPS={fps:.2f}, frames={n_frames}"

def _apply_labels_to_range(df: pd.DataFrame, behaviors: list[str], start_f: int, end_f: int, values: dict[str, int]):
    start_f = max(0, int(start_f))
    end_f = max(0, int(end_f))
    if end_f < start_f:
        start_f, end_f = end_f, start_f

    mask = (df["frame"] >= start_f) & (df["frame"] <= end_f)
    for b in behaviors:
        df.loc[mask, b] = int(values.get(b, 0))
    return df

def _merge_with_tracking(tracking_csv: str, ann_df: pd.DataFrame, out_csv: str):
    track = pd.read_csv(tracking_csv)
    if "frame" not in track.columns:
        raise KeyError("tracking_csv must have 'frame' column to merge with annotations.")
    merged = track.merge(ann_df, on="frame", how="left")
    merged.to_csv(out_csv, index=False)
    return out_csv

def render_behavior_annotator_page(video_path: str, out_csv: str):
    st.header("🎬 Behavior Annotator (Clip-based)")

    # --- init state ---
    if "clip_len" not in st.session_state:
        st.session_state.clip_len = 30  # frames
    if "clip_stride" not in st.session_state:
        st.session_state.clip_stride = 15
    if "clip_start" not in st.session_state:
        st.session_state.clip_start = 0
    if "annotations" not in st.session_state:
        st.session_state.annotations = []  # list of dict
    if "clip_version" not in st.session_state:
        st.session_state.clip_version = 0


    # --- video meta (REAL fps/frames) ---
    fps_real, n_frames, _, _ = _get_video_meta(video_path)

    # --- controls ---
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.clip_len = st.number_input(
            "Clip length (frames)", 5, 300, int(st.session_state.clip_len), step=5
        )
    with c2:
        st.session_state.clip_stride = st.number_input(
            "Stride (frames)", 1, 300, int(st.session_state.clip_stride), step=1
        )
    with c3:
        # 시간 표시용(원하면 fps_real로 고정해도 됨)
        fps_display = st.number_input("FPS (display)", 1, 240, int(round(fps_real)), step=1)

    clip_len = int(st.session_state.clip_len)
    stride = int(st.session_state.clip_stride)

    # --- clamp start/end to video length ---
    start = max(0, int(st.session_state.clip_start))
    if start >= n_frames:
        start = max(0, n_frames - clip_len)
        st.session_state.clip_start = start

    end = min(n_frames - 1, start + clip_len - 1)

    tmp_dir = os.path.join("temp", "behavior_clips")
    os.makedirs(tmp_dir, exist_ok=True)

    clip_path = os.path.abspath(os.path.join(tmp_dir, f"clip_{start}_{end}.mp4"))

    # (캐시 파일이 이상할 수 있으니, start/end 바뀌면 새로 만드는 게 안전)
    if (not os.path.exists(clip_path)) or os.path.getsize(clip_path) < 1024:
        with st.spinner("Preparing clip..."):
            clip_path = save_video_clip_ffmpeg(video_path, start, end, clip_path, fps=fps_real)

    # ✅ 디버그 정보
    st.write("clip_path:", clip_path)
    st.write("exists:", os.path.exists(clip_path))
    if os.path.exists(clip_path):
        st.write("size(bytes):", os.path.getsize(clip_path))

    # ✅ Streamlit 재생은 bytes로 주는 게 Windows/경로 문제에 강함
    with open(clip_path, "rb") as f:
        st.video(f.read(), format="video/mp4")



    
    st.write(
        f"Frames: **{start} - {end}**  |  Time: **{start/fps_display:.2f}s - {end/fps_display:.2f}s**"
    )

    # --- label UI ---
    behaviors = ["sniffing", "grooming", "rearing", "turning", "moving", "rest", "fast_moving", "other"]
    label = st.selectbox("Label for this clip", behaviors, index=0)

    colA, colB, colC, colD = st.columns(4)

    with colA:
        if st.button("✅ Save label"):
            st.session_state.annotations.append({
                "start_frame": start,
                "end_frame": end,
                "start_time_s": start / fps_real,   # ✅ 저장은 REAL fps 기준 추천
                "end_time_s": end / fps_real,
                "label": label,
            })
            st.success("Saved!")

    with colB:
        if st.button("⬅ Prev"):
            st.session_state.clip_start = max(0, start - stride)
            st.session_state.clip_version += 1
            st.rerun()

    with colC:
        if st.button("Next ➡"):
            st.session_state.clip_start = min(n_frames - 1, start + stride)
            st.session_state.clip_version += 1
            st.rerun()


    with colD:
        if st.button("💾 Export CSV"):
            df = pd.DataFrame(st.session_state.annotations)

            # ✅ out_csv가 폴더면 자동으로 파일명 붙이기
            out_csv_path = out_csv
            if out_csv_path.lower().endswith(("\\", "/")) or os.path.splitext(out_csv_path)[1] == "":
                os.makedirs(out_csv_path, exist_ok=True)
                out_csv_path = os.path.join(out_csv_path, "behavior_annotations.csv")
            else:
                os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)

            df.to_csv(out_csv_path, index=False)
            st.success(f"Saved to {out_csv_path}")

    if st.session_state.annotations:
        st.subheader("Current annotations")
        st.dataframe(pd.DataFrame(st.session_state.annotations))
