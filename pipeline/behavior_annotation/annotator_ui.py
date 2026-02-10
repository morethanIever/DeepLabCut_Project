# pipeline/behavior_annotation/annotator_ui.py
import os
import time
import shutil
import base64
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from typing import Optional, List, Dict
from pipeline.behavior_annotation.clip_utils import save_video_clip_ffmpeg

from pipeline.behavior_annotation.merge_csv import merge_labels_into_ml_features_df
from pipeline.behavior_annotation.mergeRelabelled import overwrite_labels_inplace
from pipeline.ML.randomForest import train_random_forest
from projects.projects import get_simba_labels, load_profile
from pipeline.simba_training import train_simba_models_from_annotations
try:
    from streamlit_keypress import key_press_events
except Exception:
    key_press_events = None

TEMP_CLIP_DIR = os.path.join("temp", "behavior_clips")
PROJECTS_ROOT = "projects"


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
        "annotator_video_path": "",
        "annotator_video_id": "",
        "auto_load_annotations": True,
        "annotations_loaded_for": "",
        "key_command": "",
        "last_key": "",
        "last_key_ts": 0.0,
        "replay_nonce": 0,
        "custom_labels": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


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


def _load_outlier_candidates(path: str) -> Optional[pd.DataFrame]:
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


def _is_outlier_done(outlier_frame: int, annotations: List[Dict]) -> bool:
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


def _annotation_same_window(a: Dict, b: Dict) -> bool:
    return int(a["start_frame"]) == int(b["start_frame"]) and int(a["end_frame"]) == int(b["end_frame"])


def _upsert_annotation(annotations: List[Dict], new_ann: Dict) -> List[Dict]:
    # Replace only the exact same window; keep other rows.
    kept = [a for a in annotations if not _annotation_same_window(a, new_ann)]
    kept.append(new_ann)
    kept.sort(key=lambda x: (int(x["start_frame"]), int(x["end_frame"])))
    return kept


def _normalize_annotations_df(df: pd.DataFrame, fps_real: float) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    cols = set(df.columns)
    if "label" not in cols and "behavior" in cols:
        df = df.rename(columns={"behavior": "label"})
        cols = set(df.columns)
    if "start_frame" in cols and "end_frame" in cols and "label" in cols:
        return df
    if "start" in cols and "end" in cols:
        df = df.rename(columns={"start": "start_frame", "end": "end_frame"})
        cols = set(df.columns)
        if "label" in cols:
            return df
    if "start_time_s" in cols and "end_time_s" in cols and "label" in cols:
        df["start_frame"] = (df["start_time_s"].astype(float) * fps_real).round().astype(int)
        df["end_frame"] = (df["end_time_s"].astype(float) * fps_real).round().astype(int)
        return df
    return df


def _annotation_csv_path_for_video(video_path: str) -> str:
    if not video_path:
        return ""
    v_stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(
        _get_project_dirs().get("annotations", "outputs/annotations"),
        f"{v_stem}_behavior_annotations.csv",
    )


def _load_annotations_csv(path: str, fps_real: float) -> List[Dict]:
    if not path or not os.path.exists(path):
        return []
    try:
        ann_df = pd.read_csv(path)
        ann_df = _normalize_annotations_df(ann_df, fps_real)
        required = {"start_frame", "end_frame", "label"}
        if not required.issubset(set(ann_df.columns)):
            return []
        ann_df = ann_df.sort_values(["start_frame", "end_frame"])
        return ann_df.to_dict("records")
    except Exception:
        return []


def _export_annotations_csv(video_path: str, annotations: List[Dict]) -> str:
    if not annotations:
        return ""
    ann_df = pd.DataFrame(annotations).sort_values(["start_frame", "end_frame"])
    vid_stem = os.path.splitext(os.path.basename(video_path))[0]
    return _save_df_csv(ann_df, st.session_state.ann_export_dir, f"{vid_stem}_behavior_annotations.csv")


def _get_project_dirs() -> Dict[str, str]:
    proj = st.session_state.get("active_project")
    if not proj:
        return {}
    base = os.path.join(PROJECTS_ROOT, proj, "outputs")
    return {
        "base": base,
        "videos": os.path.join(base, "videos"),
        "annotations": os.path.join(base, "annotations"),
        "ml": os.path.join(base, "ml"),
        "rf": os.path.join(base, "rf"),
    }


def _latest_ml_features_for_video(ml_dir: str, video_stem: str) -> str:
    if not ml_dir or not video_stem:
        return ""
    cand = os.path.join(ml_dir, f"{video_stem}_ml_features.csv")
    return cand if os.path.exists(cand) else ""


def _make_clip(video_path: str, start: int, end: int, fps_real: float) -> str:
    _ensure_dir(TEMP_CLIP_DIR)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    clip_path = os.path.abspath(os.path.join(TEMP_CLIP_DIR, f"{stem}_clip_{start}_{end}.mp4"))
    if (not os.path.exists(clip_path)) or os.path.getsize(clip_path) < 1024:
        save_video_clip_ffmpeg(video_path, start, end, clip_path, fps=fps_real)
    return clip_path


def _make_replay_copy(clip_path: str, nonce: int) -> str:
    """Create a replay copy to force browser reload without re-encoding."""
    if nonce <= 0:
        return clip_path
    base, ext = os.path.splitext(clip_path)
    replay_path = f"{base}_replay_{nonce}{ext}"
    if not os.path.exists(replay_path):
        shutil.copyfile(clip_path, replay_path)
    return replay_path


def _render_video_player(clip_path: str, nonce: int, video_h: int, video_w_pct: int) -> None:
    """Render a video player that autoplays and restarts on nonce change."""
    with open(clip_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("ascii")
    key_suffix = str(nonce)
    html = f"""
    <style>
      html, body {{
        margin: 0;
        padding: 0;
        background: transparent;
      }}
      .clip-wrap {{
        width: {int(video_w_pct)}%;
        margin: 0 auto;
        background: transparent;
      }}
      video {{
        width: 100%;
        height: auto;
        display: block;
        background: transparent;
      }}
    </style>
    <div class="clip-wrap">
      <video id="clip_{key_suffix}" controls autoplay>
        <source src="data:video/mp4;base64,{b64}" type="video/mp4">
      </video>
    </div>
    <script>
      const v = document.getElementById("clip_{key_suffix}");
      if (v) {{
        v.currentTime = 0;
        v.play().catch(() => {{}});
      }}
    </script>
    """
    components.html(html, height=int(video_h))


def _get_label_choices() -> List[str]:
    project_labels = []
    if st.session_state.get("active_project"):
        try:
            project_labels = get_simba_labels(PROJECTS_ROOT, st.session_state.active_project)
        except Exception:
            project_labels = []
    if project_labels:
        return project_labels
    custom = [x.strip() for x in st.session_state.get("custom_labels", "").split(",") if x.strip()]
    return custom


def _handle_key_command(
    *,
    fps_real: float,
    n_frames: int,
    clip_len: int,
    stride: int,
    video_path: str,
    label_choices: List[str],
) -> None:
    cmd = str(st.session_state.get("key_command", "")).strip().lower()
    st.session_state.key_command = ""
    if not cmd:
        return

    # Label select by number: "1".."9"
    if cmd.isdigit():
        idx = int(cmd) - 1
        if 0 <= idx < len(label_choices):
            st.session_state.clip_label_select = label_choices[idx]
        return

    start = int(st.session_state.clip_start)
    start = max(0, min(start, max(0, n_frames - clip_len)))
    end = min(n_frames - 1, start + clip_len - 1)

    if cmd in {"a", "s", "save"}:
        new_ann = {
            "start_frame": start,
            "end_frame": end,
            "start_time_s": start / fps_real,
            "end_time_s": end / fps_real,
            "label": st.session_state.get("clip_label_select"),
        }
        st.session_state.annotations = _upsert_annotation(st.session_state.annotations, new_ann)
        if hasattr(st, "toast"):
                st.toast("Saved!")
        else:
            st.success("Saved!")
        return
    if cmd in {"p", "prev"}:
        st.session_state.clip_start = max(0, start - stride)
        return
    if cmd in {"n", "next"}:
        st.session_state.clip_start = min(n_frames - 1, start + stride)
        return
    if cmd in {"f", "replay"}:
        st.session_state.replay_nonce = int(st.session_state.get("replay_nonce", 0)) + 1
        return
    if cmd in {"e", "export"}:
        out_path = _export_annotations_csv(video_path, st.session_state.annotations)
        if out_path:
            st.success(f"Saved: {out_path}")
        else:
            st.warning("No annotations to save.")


def render_behavior_annotator_page(video_path: str):
    _init_annotator_state()

    st.header("🎬 Behavior Annotator (Clip-based)")

    # Use the same video as the main page (passed in)
    proj_dirs = _get_project_dirs()
    # Prefer the current main-page video if present in session_state
    current_main_video = st.session_state.get("input_video_path")
    if current_main_video and os.path.exists(current_main_video):
        video_path = current_main_video

    if video_path and os.path.exists(video_path):
        if st.session_state.annotator_video_path != video_path:
            st.session_state.annotator_video_path = video_path
            st.session_state.annotator_video_id = os.path.basename(video_path)
            st.session_state.clip_start = 0
            st.session_state.annotations = []
        st.session_state.annotator_video_path = video_path

    st.caption(f"Annotator video: {st.session_state.annotator_video_path or '(none)'}")

    # video_path 체크
    if not video_path or (not os.path.exists(video_path)):
        st.error("No valid video loaded. 먼저 Home에서 비디오 업로드/분석 후 들어오세요.")
        return

    # Set per-project defaults once video is known
    if proj_dirs:
        st.session_state.ann_export_dir = proj_dirs["annotations"]
        st.session_state.rf_out_dir = proj_dirs["rf"]
        v_stem = os.path.splitext(os.path.basename(video_path))[0]
        auto_ml = _latest_ml_features_for_video(proj_dirs["ml"], v_stem)
        if auto_ml and not st.session_state.ml_features_csv_path:
            st.session_state.ml_features_csv_path = auto_ml

    fps_real, n_frames, _, _ = _get_video_meta(video_path)

    # Auto-load existing annotations for this video if present
    ann_csv_path = _annotation_csv_path_for_video(video_path)
    ann_id = f"{ann_csv_path}:{os.path.getmtime(ann_csv_path) if os.path.exists(ann_csv_path) else 0}"
    if (
        st.session_state.auto_load_annotations
        and (not st.session_state.annotations)
        and ann_csv_path
        and os.path.exists(ann_csv_path)
        and st.session_state.annotations_loaded_for != ann_id
    ):
        st.session_state.annotations = _load_annotations_csv(ann_csv_path, fps_real)
        st.session_state.annotations_loaded_for = ann_id

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

    # Clip player size control
    with st.expander("player size control"):
        if "clip_player_height" not in st.session_state:
            st.session_state.clip_player_height = 540
        if "clip_player_width_pct" not in st.session_state:
            st.session_state.clip_player_width_pct = 40
        st.session_state.clip_player_height = st.slider(
            "Clip player height (px)",
            min_value=240,
            max_value=1200,
            value=int(st.session_state.clip_player_height),
            step=20,
            key="clip_player_height_slider",
        )
        st.session_state.clip_player_width_pct = st.slider(
            "Clip player width (%)",
            min_value=30,
            max_value=100,
            value=int(st.session_state.clip_player_width_pct),
            step=5,
            key="clip_player_width_slider",
        )

    # Make the component container transparent (remove white background around iframe)
    st.markdown(
        """
        <style>
        div[data-testid="stIFrame"] { background: transparent !important; padding: 0 !important; margin: 0 !important; }
        div[data-testid="stIFrame"] iframe { background: transparent !important; }
        .element-container:has(iframe) { background: transparent !important; padding: 0 !important; margin: 0 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    clip_path = _make_clip(video_path, start, end, fps_real)
    clip_path = _make_replay_copy(clip_path, int(st.session_state.get("replay_nonce", 0)))
    _render_video_player(
        clip_path,
        int(st.session_state.get("replay_nonce", 0)),
        int(st.session_state.clip_player_height),
        int(st.session_state.clip_player_width_pct),
    )

    st.info(f"📍 Frames: {start}-{end} | Time: {start/fps_display:.2f}s-{end/fps_display:.2f}s")

    project_labels = []
    if st.session_state.get("active_project"):
        try:
            project_labels = get_simba_labels(PROJECTS_ROOT, st.session_state.active_project)
        except Exception:
            project_labels = []

    label_choices = _get_label_choices()
    if not project_labels:
        st.caption("Project labels are not set. You can enter custom labels below.")
        st.session_state.custom_labels = st.text_input(
            "Custom labels (comma-separated)",
            value=st.session_state.custom_labels,
            key="custom_labels_input",
        )
        label_choices = _get_label_choices()
    if not label_choices:
        st.warning("No labels available. Set project labels or enter custom labels.")
        return

    # Global keyboard shortcuts (no Enter required)
    if key_press_events is not None:
        key = key_press_events()
        if key:
            k = str(key).strip().lower()
            now = time.time()
            last_key = st.session_state.get("last_key", "")
            last_ts = float(st.session_state.get("last_key_ts", 0.0))
            if k == last_key and (now - last_ts) < 0.25:
                pass
            else:
                st.session_state.last_key = k
                st.session_state.last_key_ts = now
                if k.isdigit():
                    idx = int(k) - 1
                    if 0 <= idx < len(label_choices):
                        st.session_state.clip_label_select = label_choices[idx]
                elif k in {"s", "p", "n", "e", "f"}:
                    st.session_state.key_command = k
                    _handle_key_command(
                        fps_real=fps_real,
                        n_frames=n_frames,
                        clip_len=clip_len,
                        stride=stride,
                        video_path=video_path,
                        label_choices=label_choices,
                    )
                    if k in {"p", "n", "f"}:
                        _rerun()
    else:
        st.info("Keyboard shortcuts require `streamlit-keypress`. Install it to enable key controls.")

    label = st.selectbox("Label for this clip", label_choices, key="clip_label_select")

    colA, colB, colC, colD = st.columns(4)
    with colA:
        if st.button("Save Label"):
            new_ann = {
                "start_frame": start,
                "end_frame": end,
                "start_time_s": start / fps_real,
                "end_time_s": end / fps_real,
                "label": label,
            }
            st.session_state.annotations = _upsert_annotation(st.session_state.annotations, new_ann)
            if hasattr(st, "toast"):
                st.toast("Saved!")
            else:
                st.success("Saved!")

    with colB:
        if st.button("Prev"):
            st.session_state.clip_start = max(0, start - stride)
            _rerun()

    with colC:
        if st.button("Next"):
            st.session_state.clip_start = min(n_frames - 1, start + stride)
            _rerun()

    with colD:
        if st.button("Export Annotations CSV"):
            out_path = _export_annotations_csv(video_path, st.session_state.annotations)
            if out_path:
                st.success(f"Saved: {out_path}")
            else:
                st.warning("No annotations to save.")

    if st.session_state.annotations:
        with st.expander("Current Annotations Log", expanded=False):
            st.dataframe(pd.DataFrame(st.session_state.annotations).sort_values(["start_frame", "end_frame"]))

    st.markdown("**Keyboard Controls**")
    st.caption("Press keys (no Enter): `s`=save, `p`=prev, `n`=next, `f`=replay, `e`=export, `1..9`=select label. You can also type below and press Enter.")
   

    with st.expander("Annotation Storage", expanded=False):
        st.checkbox("Auto-load annotations from CSV when available", value=st.session_state.auto_load_annotations, key="auto_load_annotations")
        if st.button("Load annotations from CSV"):
            ann_csv_path = _annotation_csv_path_for_video(video_path)
            loaded = _load_annotations_csv(ann_csv_path, fps_real)
            if loaded:
                st.session_state.annotations = loaded
                st.session_state.annotations_loaded_for = f"{ann_csv_path}:{os.path.getmtime(ann_csv_path)}"
                st.success(f"Loaded: {ann_csv_path}")
            else:
                st.warning("No valid annotation CSV found.")


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
            _rerun()

        if sc2.button("🎯 Next Outlier"):
            st.session_state.outlier_idx = (idx + 1) % len(outliers)
            st.session_state.clip_start = int(outliers.iloc[st.session_state.outlier_idx]["clip_start"])
            _rerun()

        if st.sidebar.button("💾 Export Outlier-only CSV"):
            if outliers is None or "frame" not in outliers.columns:
                st.sidebar.error("Outlier CSV is missing or invalid (no 'frame' column).")
                return
            out_frames = set(outliers["frame"].astype(int))
            kept = []
            anns = st.session_state.annotations
            if not anns:
                ann_path = _annotation_csv_path_for_video(video_path)
                if os.path.exists(ann_path):
                    anns = _load_annotations_csv(ann_path, fps_real)
            for a in anns:
                s = int(a["start_frame"]); e = int(a["end_frame"])
                if any(f in out_frames for f in range(s, e + 1)):
                    kept.append(a)
            if not kept:
                st.sidebar.warning("No outlier annotations found in memory.")
            else:
                v_stem = os.path.splitext(os.path.basename(video_path))[0]
                out_path = _save_df_csv(
                    pd.DataFrame(kept),
                    _get_project_dirs().get("annotations", "outputs/annotations"),
                    f"{v_stem}_behavior_annotations_outliers.csv",
                )
                st.sidebar.success(f"Saved: {out_path}")

    if st.sidebar.button("⚡ Create Outlier CSV from Annotations"):
        v_stem = os.path.splitext(os.path.basename(video_path))[0]
        ann_path = os.path.join(
            _get_project_dirs().get("annotations", "outputs/annotations"),
            f"{v_stem}_behavior_annotations.csv"
        )
        if not os.path.exists(ann_path):
            st.sidebar.error("Annotation CSV not found for this video.")
        else:
            ann_df = pd.read_csv(ann_path)
            ann_df = _normalize_annotations_df(ann_df, fps_real)
            required = {"start_frame", "end_frame"}
            if not required.issubset(set(ann_df.columns)):
                st.sidebar.error(f"Annotations must contain columns: {sorted(required)}")
            else:
                outlier_frames = ((ann_df["start_frame"].astype(int) + ann_df["end_frame"].astype(int)) // 2)
                out_df = pd.DataFrame({"frame": outlier_frames})
                out_path = os.path.join(
                    _get_project_dirs().get("annotations", "outputs/annotations"),
                    f"{v_stem}_outliers.csv"
                )
                out_df.to_csv(out_path, index=False)
                st.sidebar.success(f"Saved: {out_path}")

    # -----------------------
    # 아래: 모델 파이프라인 
    # -----------------------
    st.markdown("---")
    st.subheader("🧠 Model Pipeline")

    st.subheader("1) Initial Train")
    st.write("행동 라벨링 annotation.csv + ml_feature.csv Merge -> RandomForest 학습")

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
        if st.button("🔗 Merge (features + annotations)", type="primary"):
            if not st.session_state.ml_features_csv_path or (not os.path.exists(st.session_state.ml_features_csv_path)):
                st.error("ml_features.csv path is invalid.")
            else:
                ann_df = pd.DataFrame(st.session_state.annotations)
                if ann_df.empty:
                    ann_path = _annotation_csv_path_for_video(video_path)
                    if os.path.exists(ann_path):
                        ann_df = pd.read_csv(ann_path)
                ann_df = _normalize_annotations_df(ann_df, fps_real)
                required = {"start_frame", "end_frame", "label"}
                if not required.issubset(set(ann_df.columns)):
                    st.error(f"Annotations must contain columns: {sorted(required)}")
                    st.stop()
                feats_df = pd.read_csv(st.session_state.ml_features_csv_path)

                merged_df = merge_labels_into_ml_features_df(
                    feats_df,
                    ann_df,
                    mode="onehot",
                    boundary_exclude=int(boundary_exclude),
                    behaviors=label_choices,
                )

                _ensure_dir(st.session_state.rf_out_dir)
                merged_path = os.path.join(st.session_state.rf_out_dir, "train_merged.csv")
                merged_df.to_csv(merged_path, index=False)
                st.session_state.merged_csv_path = merged_path

                st.success(f"Merged saved: {merged_path}")
                st.dataframe(merged_df.head(50))

    with cB:
        if st.button("🟢 Train RandomForest (from merged)", type="primary"):
            if not st.session_state.merged_csv_path or (not os.path.exists(st.session_state.merged_csv_path)):
                st.error("No merged_csv found. Merge를 먼저 실행")
            else:
                _ensure_dir(st.session_state.rf_out_dir)
                train_random_forest(
                    st.session_state.merged_csv_path,
                    st.session_state.rf_out_dir,
                    behaviors=label_choices,
                    use_time_split=True,
                    use_train_mask=True,
                )
                st.success("RF trained! outputs/rf 에 모델/리포트 저장됨")

    st.markdown("---")
    st.subheader("2) Outlier Relabel & Retrain")
    st.write("Outlier에서 다시 라벨링한 결과 -> 기존 merged csv에 덮어쓴 후 재학습")

    relabel_csv = st.text_input(
        "Outlier relabel CSV path (behavior_annotations_outliers.csv)",
        value=os.path.join(st.session_state.ann_export_dir, "behavior_annotations_outliers.csv"),
        key="relabel_csv_input"
    )

    overwrite_boundary = st.number_input("Outlier boundary exclude frames", 0, 30, 5, step=1, key="boundary_exclude_outlier")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("🧩 Apply Outlier Relabel (overwrite merged)", type="primary"):
            if not st.session_state.merged_csv_path or (not os.path.exists(st.session_state.merged_csv_path)):
                st.error("No merged_csv found. Initial Train 탭에서 Merge 먼저 해주세요.")
            elif not relabel_csv or (not os.path.exists(relabel_csv)):
                st.error("Outlier relabel CSV path is invalid.")
            else:
                base_csv = st.session_state.merged_csv_path
                updated_csv = base_csv.replace(".csv", "_updated.csv")

                overwrite_labels_inplace(
                    base_csv=base_csv,
                    relabel_csv=relabel_csv,
                    out_csv=updated_csv,
                    behaviors=label_choices,
                    boundary_exclude=int(overwrite_boundary),
                    tie_break="last",
                    update_train_mask=True,
                )

                st.session_state.merged_csv_path = updated_csv
                st.success(f"Updated merged saved: {updated_csv}")

    with c2:
        if st.button("🔁 Re-train RandomForest (updated merged)", type="primary"):
            if not st.session_state.merged_csv_path or (not os.path.exists(st.session_state.merged_csv_path)):
                st.error("No updated merged_csv found. Outlier overwrite 먼저 하기")
            else:
                _ensure_dir(st.session_state.rf_out_dir)
                train_random_forest(
                    st.session_state.merged_csv_path,
                    st.session_state.rf_out_dir,
                    behaviors=label_choices,
                    use_time_split=True,
                    use_train_mask=True,
                )
                st.success("RF re-trained with outlier relabels!")

    st.markdown("---")
    st.subheader("3) Train SimBA Models (Per-Project)")
    st.write("Train one binary SimBA model per behavior label using SimBA features_extracted.")

    if st.button("🧪 Train SimBA Models", type="primary"):
        try:
            if not st.session_state.get("active_project"):
                st.error("No active project.")
            else:
                prof = load_profile(PROJECTS_ROOT, st.session_state.active_project)
                cfg_path = prof.get("simba", {}).get("config_path", "")
                if not cfg_path or not os.path.exists(cfg_path):
                    st.error("SimBA config not found. Set it in Project Setup.")
                else:
                    v_stem = os.path.splitext(os.path.basename(video_path))[0]
                    ann_path = os.path.join(
                        _get_project_dirs().get("annotations", "outputs/annotations"),
                        f"{v_stem}_behavior_annotations.csv"
                    )
                    if not os.path.exists(ann_path) and st.session_state.annotations:
                        ann_df = pd.DataFrame(st.session_state.annotations)
                        ann_path = _save_df_csv(
                            ann_df,
                            _get_project_dirs().get("annotations", "outputs/annotations"),
                            f"{v_stem}_behavior_annotations.csv",
                        )
                    if os.path.exists(ann_path):
                        ann_df = pd.read_csv(ann_path)
                        ann_df = _normalize_annotations_df(ann_df, fps_real)
                        required = {"start_frame", "end_frame", "label"}
                        if not required.issubset(set(ann_df.columns)):
                            st.error(f"Annotations must contain columns: {sorted(required)}")
                            st.stop()
                        ann_path = _save_df_csv(
                            ann_df,
                            _get_project_dirs().get("annotations", "outputs/annotations"),
                            f"{v_stem}_behavior_annotations.csv",
                        )
                    labels = get_simba_labels(PROJECTS_ROOT, st.session_state.active_project)
                    if not labels:
                        st.error("No project labels set. Save labels in Project Setup.")
                    else:
                        saved = train_simba_models_from_annotations(
                            cfg_path,
                            annotations_csv=ann_path,
                            labels=labels,
                            video_stem=v_stem,
                            threshold=0.5,
                            min_bout=5,
                        )
                        st.success(f"Trained {len(saved)} SimBA models.")
        except Exception as e:
            st.error(f"SimBA training failed: {e}")
