# pipeline/behavior_annotation/annotator_ui.py
import os
import json
from pathlib import Path
import time
import shutil
import base64
import configparser
import cv2
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
from typing import Optional, List, Dict
import subprocess
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
        "key_buffer": "",
        "key_buffer_ts": 0.0,
        "replay_nonce": 0,
        "custom_labels": "",
        "last_outlier_csv_path": "",
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


def _load_outlier_candidates(path: str, *, fps_real: float) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        # Normalize various outlier formats to include frame/clip_start/clip_end
        if "frame" not in df.columns:
            if "start_frame" in df.columns and "end_frame" in df.columns:
                df["frame"] = df["start_frame"].astype(int)
            elif "start_sec" in df.columns and "end_sec" in df.columns and fps_real > 0:
                df["start_frame"] = (df["start_sec"].astype(float) * fps_real).round().astype(int)
                df["end_frame"] = (df["end_sec"].astype(float) * fps_real).round().astype(int)
                df["frame"] = df["start_frame"].astype(int)
            else:
                return None
        if "clip_start" not in df.columns:
            if "start_frame" in df.columns:
                df["clip_start"] = df["start_frame"].astype(int)
            else:
                df["clip_start"] = (df["frame"].astype(int) - 15).clip(lower=0)
        if "clip_end" not in df.columns:
            if "end_frame" in df.columns:
                df["clip_end"] = df["end_frame"].astype(int)
            else:
                df["clip_end"] = df["clip_start"].astype(int) + 29
        return df
    except Exception:
        return None

def _is_outlier_done(outlier_frame: int, annotations: List[Dict]) -> bool:
    return any(int(a["start_frame"]) <= outlier_frame <= int(a["end_frame"]) for a in annotations)


def _make_outliers_from_machine_results(
    machine_csv: str,
    *,
    threshold: float,
    clip_len: int,
    pre_frames: int,
) -> pd.DataFrame:
    df = pd.read_csv(machine_csv)
    prob_cols = [c for c in df.columns if c.lower().startswith("probability_") or c.lower().startswith("prob_")]
    if not prob_cols:
        raise ValueError("No probability columns found in machine_results.")

    probs = df[prob_cols].astype(float)
    max_prob = probs.max(axis=1)
    pred_idx = probs.values.argmax(axis=1)
    pred_label = [prob_cols[i] for i in pred_idx]

    if "frame" in df.columns:
        frame = df["frame"].astype(int)
    else:
        frame = pd.Series(np.arange(len(df), dtype=int), name="frame")

    mask = max_prob < float(threshold)
    out = pd.DataFrame({
        "frame": frame[mask].values,
        "pred_label": np.array(pred_label)[mask],
        "conf": max_prob[mask].values,
    })
    if out.empty:
        return out

    pre_frames = int(pre_frames)
    clip_len = int(clip_len)
    out["clip_start"] = (out["frame"].astype(int) - pre_frames).clip(lower=0)
    out["clip_end"] = out["clip_start"].astype(int) + max(1, clip_len) - 1
    return out


def _make_outliers_from_rf_model(
    merged_csv: str,
    model_path: str,
    meta_path: str,
    *,
    threshold: float,
    clip_len: int,
    pre_frames: int,
    stride: int,
) -> pd.DataFrame:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"RF model not found: {model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"RF meta not found: {meta_path}")

    meta = joblib.load(meta_path)
    feature_cols = list(meta.get("feature_columns", []))
    class_names = list(meta.get("class_names", []))
    if not feature_cols:
        raise RuntimeError("RF meta is missing feature_columns.")
    if not class_names:
        raise RuntimeError("RF meta is missing class_names.")

    df = pd.read_csv(merged_csv)
    if "frame" in df.columns:
        frame = df["frame"].astype(int)
    else:
        frame = pd.Series(np.arange(len(df), dtype=int), name="frame")

    # Build X with the same columns used in training
    X = pd.DataFrame(index=df.index)
    for c in feature_cols:
        if c in df.columns:
            X[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            X[c] = 0.0
    X = X.fillna(0.0)

    clf = joblib.load(model_path)
    probs = clf.predict_proba(X)
    max_prob = probs.max(axis=1)
    pred_idx = probs.argmax(axis=1)
    pred_label = [class_names[i] for i in pred_idx]

    max_prob = np.asarray(max_prob)
    mask = max_prob < float(threshold)
    out = pd.DataFrame({
        "frame": frame[mask].values,
        "pred_label": np.array(pred_label)[mask],
        "conf": max_prob[mask],
    })
    if out.empty:
        return out

    pre_frames = int(pre_frames)
    clip_len = int(clip_len)
    stride = max(1, int(stride))
    clip_start = (out["frame"].astype(int) - pre_frames).clip(lower=0)
    # snap to stride grid so labeling windows align
    clip_start = (clip_start // stride) * stride
    out["clip_start"] = clip_start.astype(int)
    out["clip_end"] = out["clip_start"].astype(int) + max(1, clip_len) - 1

    # Keep one row per clip window (lowest confidence = most uncertain)
    out = out.sort_values("conf", ascending=True).drop_duplicates(subset=["clip_start"], keep="first")
    out = out.sort_values("clip_start").reset_index(drop=True)
    return out


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


def _render_rf_diagnostics(out_dir: str, *, max_features: int = 20) -> None:
    if not out_dir:
        return
    cm_path = os.path.join(out_dir, "val_confusion_matrix.csv")
    fi_path = os.path.join(out_dir, "feature_importance.csv")
    report_path = os.path.join(out_dir, "val_report.txt")

    if os.path.exists(cm_path):
        try:
            cm_df = pd.read_csv(cm_path, index_col=0)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm_df.values, cmap="Blues")
            ax.set_title("Validation Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_xticks(range(len(cm_df.columns)))
            ax.set_yticks(range(len(cm_df.index)))
            ax.set_xticklabels(cm_df.columns, rotation=45, ha="right")
            ax.set_yticklabels(cm_df.index)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.warning(f"Failed to render confusion matrix: {e}")
    else:
        st.info("val_confusion_matrix.csv not found yet.")

    if os.path.exists(fi_path):
        try:
            fi_df = pd.read_csv(fi_path)
            if not fi_df.empty:
                top = fi_df.sort_values("importance", ascending=False).head(max_features)
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.barh(top["feature"][::-1], top["importance"][::-1])
                ax.set_title(f"Feature Importance (Top {len(top)})")
                ax.set_xlabel("Importance")
                ax.set_ylabel("Feature")
                st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.warning(f"Failed to render feature importance: {e}")
    else:
        st.info("feature_importance.csv not found yet.")

    if os.path.exists(report_path):
        try:
            report_txt = Path(report_path).read_text(encoding="utf-8")
            st.markdown("**Validation Report**")
            st.code(report_txt, language="text")            
        except Exception as e:
            st.warning(f"Failed to load val_report.txt: {e}")
    else:
        st.info("val_report.txt not found yet.")


def _run_simba_feature_extraction(cfg_path: str, *, conda_env: str = "simba_fast") -> subprocess.CompletedProcess:
    conda_exe = os.environ.get("CONDA_EXE")
    if not conda_exe:
        candidate = os.path.join(os.path.expanduser("~"), "anaconda3", "Scripts", "conda.exe")
        if os.path.exists(candidate):
            conda_exe = candidate
        else:
            conda_exe = "conda"

    cmd = [
        conda_exe, "run", "-n", conda_env,
        "python", "-c",
        (
            "from simba.utils.cli.cli_tools import feature_extraction_runner; "
            f"feature_extraction_runner(config_path=r'{cfg_path}')"
        ),
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def _ensure_simba_pose_setting(config_path: str) -> str:
    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")
    if "create ensemble settings" not in cfg:
        cfg["create ensemble settings"] = {}
    pose_setting = cfg["create ensemble settings"].get("pose_estimation_body_parts", "").strip()
    if pose_setting:
        return pose_setting

    proj_path = cfg.get("General settings", "project_path", fallback="").strip()
    animal_cnt_raw = cfg.get("General settings", "animal_no", fallback="1").strip()
    try:
        animal_cnt = int(animal_cnt_raw)
    except Exception:
        animal_cnt = 1

    bp_cnt = None
    if proj_path:
        bp_path = os.path.join(
            proj_path,
            "logs",
            "measures",
            "pose_configs",
            "bp_names",
            "project_bp_names.csv",
        )
        if os.path.exists(bp_path):
            try:
                bp_df = pd.read_csv(bp_path, header=None)
                bp_cnt = int(bp_df.iloc[:, 0].dropna().shape[0])
            except Exception:
                bp_cnt = None

    # Map to supported SimBA presets or fall back to user_defined
    supported_1 = {4, 7, 8, 9}
    supported_2 = {8, 14, 16}
    if animal_cnt == 1 and bp_cnt in supported_1:
        pose_setting = str(bp_cnt)
    elif animal_cnt == 2 and bp_cnt in supported_2:
        pose_setting = str(bp_cnt)
    else:
        pose_setting = "user_defined"

    cfg["create ensemble settings"]["pose_estimation_body_parts"] = pose_setting
    with open(config_path, "w", encoding="utf-8") as f:
        cfg.write(f)
    return pose_setting


def _run_simba_train_models(
    *,
    config_path: str,
    annotations_csv: str,
    labels: List[str],
    video_stem: str,
    video_path: str | None,
    merged_csv_fallback: str | None,
    threshold: float,
    min_bout: int,
    conda_env: str = "simba_fast",
) -> subprocess.CompletedProcess:
    conda_exe = os.environ.get("CONDA_EXE")
    if not conda_exe:
        candidate = os.path.join(os.path.expanduser("~"), "anaconda3", "Scripts", "conda.exe")
        if os.path.exists(candidate):
            conda_exe = candidate
        else:
            conda_exe = "conda"

    env = os.environ.copy()
    env.update(
        {
            "SIMBA_CFG_PATH": str(config_path),
            "SIMBA_ANN_CSV": str(annotations_csv),
            "SIMBA_LABELS": json.dumps(labels),
            "SIMBA_VIDEO_STEM": str(video_stem),
            "SIMBA_VIDEO_PATH": str(video_path or ""),
            "SIMBA_MERGED_FALLBACK": str(merged_csv_fallback or ""),
            "SIMBA_THRESHOLD": str(threshold),
            "SIMBA_MIN_BOUT": str(min_bout),
        }
    )
    code = (
        "import os, json; "
        "from pipeline.simba_training import train_simba_models_from_annotations as f; "
        "labels=json.loads(os.environ['SIMBA_LABELS']); "
        "saved=f(os.environ['SIMBA_CFG_PATH'], "
        "annotations_csv=os.environ['SIMBA_ANN_CSV'], "
        "labels=labels, "
        "video_stem=os.environ['SIMBA_VIDEO_STEM'], "
        "video_path=(os.environ.get('SIMBA_VIDEO_PATH') or None), "
        "merged_csv_fallback=(os.environ.get('SIMBA_MERGED_FALLBACK') or None), "
        "threshold=float(os.environ.get('SIMBA_THRESHOLD','0.5')), "
        "min_bout=int(os.environ.get('SIMBA_MIN_BOUT','5'))); "
        "print(json.dumps({'ok': True, 'saved': saved}))"
    )
    cmd = [conda_exe, "run", "-n", conda_env, "python", "-c", code]
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def _run_simba_train_models_multi(
    *,
    config_path: str,
    annotations_csvs: List[str],
    video_stems: List[str],
    labels: List[str],
    merged_csv_fallback: str | None,
    threshold: float,
    min_bout: int,
    conda_env: str = "simba_fast",
) -> subprocess.CompletedProcess:
    conda_exe = os.environ.get("CONDA_EXE")
    if not conda_exe:
        candidate = os.path.join(os.path.expanduser("~"), "anaconda3", "Scripts", "conda.exe")
        if os.path.exists(candidate):
            conda_exe = candidate
        else:
            conda_exe = "conda"

    env = os.environ.copy()
    env.update(
        {
            "SIMBA_CFG_PATH": str(config_path),
            "SIMBA_ANN_LIST": json.dumps(annotations_csvs),
            "SIMBA_VIDEO_STEMS": json.dumps(video_stems),
            "SIMBA_LABELS": json.dumps(labels),
            "SIMBA_MERGED_FALLBACK": str(merged_csv_fallback or ""),
            "SIMBA_THRESHOLD": str(threshold),
            "SIMBA_MIN_BOUT": str(min_bout),
        }
    )
    code = (
        "import os, json; "
        "from pipeline.simba_training import train_simba_models_from_annotations_multi as f; "
        "anns=json.loads(os.environ['SIMBA_ANN_LIST']); "
        "stems=json.loads(os.environ['SIMBA_VIDEO_STEMS']); "
        "labels=json.loads(os.environ['SIMBA_LABELS']); "
        "saved=f(os.environ['SIMBA_CFG_PATH'], "
        "annotations_csvs=anns, "
        "labels=labels, "
        "video_stems=stems, "
        "merged_csv_fallback=(os.environ.get('SIMBA_MERGED_FALLBACK') or None), "
        "threshold=float(os.environ.get('SIMBA_THRESHOLD','0.5')), "
        "min_bout=int(os.environ.get('SIMBA_MIN_BOUT','5'))); "
        "print(json.dumps({'ok': True, 'saved': saved}))"
    )
    cmd = [conda_exe, "run", "-n", conda_env, "python", "-c", code]
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


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

    # Label select by number (supports multi-digit, e.g. "12" -> label 12)
    if cmd.isdigit():
        idx = int(cmd) - 1
        if 0 <= idx < len(label_choices):
            st.session_state.clip_label_select = label_choices[idx]
        return

    start = int(st.session_state.clip_start)
    start = max(0, min(start, max(0, n_frames - clip_len)))
    end = min(n_frames - 1, start + clip_len - 1)

    # Outlier mode navigation (if outlier list is loaded)
    outlier_mode = bool(st.session_state.get("outlier_mode_active"))
    outlier_rows = st.session_state.get("outlier_rows") or []


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
        if outlier_mode and outlier_rows:
            st.session_state.outlier_idx = (int(st.session_state.get("outlier_idx", 0)) - 1) % len(outlier_rows)
            st.session_state.clip_start = int(outlier_rows[st.session_state.outlier_idx]["clip_start"])
        else:
            st.session_state.clip_start = max(0, start - stride)
        return
    if cmd in {"n", "next"}:
        if outlier_mode and outlier_rows:
            st.session_state.outlier_idx = (int(st.session_state.get("outlier_idx", 0)) + 1) % len(outlier_rows)
            st.session_state.clip_start = int(outlier_rows[st.session_state.outlier_idx]["clip_start"])
        else:
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


def render_behavior_annotator_page(video_path: str, *, page_mode: str = "labeling"):
    _init_annotator_state()

    mode = str(page_mode or "labeling").lower()
    show_clip_ui = mode in {"labeling", "outlier"}
    show_outlier_sidebar = mode == "outlier"
    show_initial_train = mode == "labeling"
    show_outlier_retrain = mode == "outlier"
    show_simba_train = mode == "simba"

    if mode == "labeling":
        st.header("🧩 Behavior Labeling + Initial Train")
    elif mode == "outlier":
        st.header("🎯 Outlier Relabel + Retrain")
    elif mode == "simba":
        st.header("🧪 Train SimBA Models")
    else:
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
            st.session_state.annotations_loaded_for = ""
            st.session_state.outlier_mode_active = False
            st.session_state.outlier_rows = []
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

    # Preload outlier list early so main navigation can respect it
    if show_outlier_retrain and st.session_state.get("outlier_csv_path"):
        preload = _load_outlier_candidates(
            st.session_state.get("outlier_csv_path"),
            fps_real=fps_real,
        )
        if preload is not None and len(preload) > 0:
            st.session_state.outlier_mode_active = True
            st.session_state.outlier_rows = preload.to_dict("records")

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
            # Longer cooldown for navigation keys to prevent repeat/hold from skipping clips
            cooldown = 0.6 if k in {"p", "n", "f"} else 0.25
            if k == last_key and (now - last_ts) < cooldown:
                pass
            else:
                st.session_state.last_key = k
                st.session_state.last_key_ts = now
                if k.isdigit():
                    # Support 10+ labels by buffering multi-digit input (e.g., "1" then "2" -> label 12)
                    buf = str(st.session_state.get("key_buffer", ""))
                    buf_ts = float(st.session_state.get("key_buffer_ts", 0.0))
                    # Reset buffer if too slow between digits
                    if (now - buf_ts) > 0.7:
                        buf = ""
                    buf = f"{buf}{k}"
                    st.session_state.key_buffer = buf
                    st.session_state.key_buffer_ts = now
                    idx = int(buf) - 1
                    if 0 <= idx < len(label_choices):
                        st.session_state.clip_label_select = label_choices[idx]
                        # Clear buffer after a successful multi-digit selection
                        st.session_state.key_buffer = ""
                        st.session_state.key_buffer_ts = 0.0
                elif k in {"s", "p", "n", "e", "f"}:
                    # Clear digit buffer when issuing a command
                    st.session_state.key_buffer = ""
                    st.session_state.key_buffer_ts = 0.0
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
            outlier_mode = bool(st.session_state.get("outlier_mode_active"))
            outlier_rows = st.session_state.get("outlier_rows") or []
            if outlier_mode and outlier_rows:
                st.session_state.outlier_idx = (int(st.session_state.get("outlier_idx", 0)) - 1) % len(outlier_rows)
                st.session_state.clip_start = int(outlier_rows[st.session_state.outlier_idx]["clip_start"])
            else:
                st.session_state.clip_start = max(0, start - stride)
            _rerun()

    with colC:
        if st.button("Next"):
            outlier_mode = bool(st.session_state.get("outlier_mode_active"))
            outlier_rows = st.session_state.get("outlier_rows") or []
            if outlier_mode and outlier_rows:
                st.session_state.outlier_idx = (int(st.session_state.get("outlier_idx", 0)) + 1) % len(outlier_rows)
                st.session_state.clip_start = int(outlier_rows[st.session_state.outlier_idx]["clip_start"])
            else:
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
    st.caption("Press keys (no Enter): `s`=save, `p`=prev, `n`=next, `f`=replay, `e`=export, digits=`1..99` select label (type quickly for 10+). You can also type below and press Enter.")
   

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
    # Outlier Mode (main page)
    # -----------------------
    if show_outlier_retrain:
        st.markdown("---")
        st.subheader("Outlier CSV from RF Model")
        with st.expander("Export outlier csv"):
            proj_dirs = _get_project_dirs()
            rf_dir = proj_dirs.get("rf", "outputs/rf")

            col_rf_a, col_rf_b = st.columns(2)
            with col_rf_a:
                rf_model_path = st.text_input(
                    "RF model path",
                    value=os.path.join(rf_dir, "rf_model.joblib"),
                    key="rf_model_path_input",
                )
                rf_meta_path = st.text_input(
                    "RF meta path",
                    value=os.path.join(rf_dir, "rf_meta.joblib"),
                    key="rf_meta_path_input",
                )
            with col_rf_b:
                rf_merged_csv = st.text_input(
                    "RF train_merged CSV path",
                    value=os.path.join(rf_dir, "train_merged.csv"),
                    key="rf_train_merged_input",
                )
                rf_thresh = st.number_input(
                    "RF low-confidence threshold", 0.0, 1.0, 0.5, step=0.05, key="rf_outlier_thresh"
                )
                rf_clip_len = st.number_input(
                    "RF clip length (frames)", 5, 300, 30, step=1, key="rf_outlier_clip_len"
                )
                rf_pre_frames = st.number_input(
                    "RF pre frames", 0, 150, 15, step=1, key="rf_outlier_pre_frames"
                )

        if st.button("Create Outlier CSV from RF Model"):
            if not rf_merged_csv or not os.path.exists(rf_merged_csv):
                st.error("train_merged CSV path is invalid.")
            else:
                try:
                    out_df = _make_outliers_from_rf_model(
                        rf_merged_csv,
                        rf_model_path,
                        rf_meta_path,
                        threshold=float(rf_thresh),
                        clip_len=int(rf_clip_len),
                        pre_frames=int(rf_pre_frames),
                        stride=int(st.session_state.get("clip_stride", 15)),
                    )
                    if out_df.empty:
                        st.warning("No RF outliers found with current threshold.")
                    else:
                        v_stem = os.path.splitext(os.path.basename(rf_merged_csv))[0]
                        out_path = os.path.join(
                            _get_project_dirs().get("annotations", "outputs/annotations"),
                            f"{v_stem}_outliers_rf.csv"
                        )
                        out_df.to_csv(out_path, index=False)
                        st.success(f"Saved: {out_path}")
                        st.session_state.outlier_csv_path = out_path
                except Exception as e:
                    st.error(f"Failed: {e}")

        st.markdown("---")
        st.subheader("Outlier Review")
        st.session_state.outlier_csv_path = st.text_input(
            "Outlier CSV path",
            value=st.session_state.outlier_csv_path,
            key="outlier_csv_path_input"
        )
        if st.session_state.outlier_csv_path and st.session_state.outlier_csv_path != st.session_state.get("last_outlier_csv_path"):
            st.session_state.last_outlier_csv_path = st.session_state.outlier_csv_path
            st.session_state.annotations = []
            st.session_state.annotations_loaded_for = ""
            st.session_state.outlier_idx = 0

        outliers = _load_outlier_candidates(st.session_state.outlier_csv_path, fps_real=fps_real)
        if outliers is not None and len(outliers) > 0:
            # expose outlier rows for keyboard navigation
            st.session_state.outlier_mode_active = True
            st.session_state.outlier_rows = outliers.to_dict("records")
            outliers["done"] = outliers["frame"].apply(lambda f: _is_outlier_done(int(f), st.session_state.annotations))
            n_done = int(outliers["done"].sum())
            st.progress(n_done / len(outliers), text=f"Progress: {n_done}/{len(outliers)}")
            
            idx = int(st.session_state.outlier_idx) % len(outliers)
            cur = outliers.iloc[idx]

            if "pred_label" in outliers.columns:
                st.info(f"Model Suggestion: {cur.get('pred_label')} (conf={cur.get('conf', 0):.2f})")
                nav1, nav2, nav3 = st.columns([1, 1, 2])
                with nav1:
                    if st.button("Prev Outlier"):
                        st.session_state.outlier_idx = (idx - 1) % len(outliers)
                        st.session_state.clip_start = int(outliers.iloc[st.session_state.outlier_idx]["clip_start"])
                        _rerun()
                with nav2:
                    if st.button("Next Outlier"):
                        st.session_state.outlier_idx = (idx + 1) % len(outliers)
                        st.session_state.clip_start = int(outliers.iloc[st.session_state.outlier_idx]["clip_start"])
                        _rerun()
                with nav3:
                    if st.button("Export Outlier-only CSV"):
                        if outliers is None:
                            st.error("Outlier CSV is missing or invalid.")
                            return
                        if "start_frame" in outliers.columns and "end_frame" in outliers.columns:
                            out_windows = list(zip(outliers["start_frame"].astype(int), outliers["end_frame"].astype(int)))
                        else:
                            out_frames = set(outliers["frame"].astype(int))
                            out_windows = [(f, f) for f in out_frames]
                        kept = []
                        anns = st.session_state.annotations
                        if not anns:
                            ann_path = _annotation_csv_path_for_video(video_path)
                            if os.path.exists(ann_path):
                                anns = _load_annotations_csv(ann_path, fps_real)
                        for a in anns:
                            s = int(a["start_frame"]); e = int(a["end_frame"])
                            if any((s <= ow_e and e >= ow_s) for ow_s, ow_e in out_windows):
                                kept.append(a)
                        if not kept:
                            st.warning("No outlier annotations found in memory.")
                        else:
                            v_stem = os.path.splitext(os.path.basename(video_path))[0]
                            out_path = _save_df_csv(
                                pd.DataFrame(kept),
                                _get_project_dirs().get("annotations", "outputs/annotations"),
                                f"{v_stem}_behavior_annotations_outliers.csv",
                            )
                            st.success(f"Saved: {out_path}")
            else:
                st.session_state.outlier_mode_active = False
                st.session_state.outlier_rows = []


    if show_initial_train or show_outlier_retrain or show_simba_train:
        st.markdown("---")
        st.subheader("🧠 Model Pipeline")

    if show_initial_train:
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

        v_stem = os.path.splitext(os.path.basename(video_path))[0]
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
                    merged_path = os.path.join(st.session_state.rf_out_dir, f"train_merged_{v_stem}.csv")
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

        st.markdown("**RF Diagnostics**")
        _render_rf_diagnostics(st.session_state.rf_out_dir, max_features=20)

    if show_outlier_retrain:
        st.markdown("---")
        st.subheader("2) Outlier Relabel & Retrain")
        st.write("Relabeled annotations are merged into a per-video train_merged_<video>.csv, then retrained.")

        v_stem = os.path.splitext(os.path.basename(video_path))[0]
        default_relabel = os.path.join(
            _get_project_dirs().get("annotations", "outputs/annotations"),
            f"{v_stem}_behavior_annotations.csv",
        )
        if not os.path.exists(default_relabel):
            default_relabel = os.path.join(
                _get_project_dirs().get("annotations", "outputs/annotations"),
                "behavior_annotations_outliers.csv",
            )


        relabel_csv = st.text_input(
            "Outlier relabel CSV path",
            value=default_relabel,
            key="relabel_csv_input"
        )

        merged_csv_input = st.text_input(
            "Merged CSV path (train_merged_<video>.csv)",
            value=os.path.join(st.session_state.rf_out_dir, f"train_merged_{v_stem}.csv"),
            key="merged_csv_path_input",
        )

        overwrite_boundary = st.number_input("Outlier boundary exclude frames", 0, 30, 0, step=1, key="boundary_exclude_outlier")

        c1, c2 = st.columns(2)

        with c1:
            if st.button("🧩 Apply Outlier Relabel (overwrite merged)", type="primary"):
                base_csv = os.path.join(st.session_state.rf_out_dir, f"train_merged_{v_stem}.csv")
                if not os.path.exists(base_csv):
                    st.error("No merged_csv found. Initial Train 탭에서 Merge 먼저 해주세요.")
                elif not relabel_csv or (not os.path.exists(relabel_csv)):
                    st.error("Outlier relabel CSV path is invalid.")
                else:
                    updated_csv = os.path.join(st.session_state.rf_out_dir, f"train_merged_{v_stem}_updated.csv")

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

        st.markdown("**RF Diagnostics**")
        _render_rf_diagnostics(st.session_state.rf_out_dir, max_features=20)

    if show_simba_train:
        st.markdown("---")
        st.subheader("3) Train SimBA Models (Per-Project)")
        st.write("Train one binary SimBA model per behavior label using SimBA features_extracted.")

        st.markdown("**SimBA Training Preflight**")
        try:
            project_path = None
            cfg_path = ""
            if st.session_state.get("active_project"):
                project_path = os.path.join(PROJECTS_ROOT, st.session_state.active_project, "simba", "project_folder")
                prof = load_profile(PROJECTS_ROOT, st.session_state.active_project)
                cfg_path = prof.get("simba", {}).get("config_path", "")
            cfg = configparser.ConfigParser()
            if cfg_path and os.path.exists(cfg_path):
                cfg.read(cfg_path, encoding="utf-8")
            else:
                st.warning("SimBA config not found. Set it in Project Setup.")
                cfg_path = ""

            desired_pose_dir = os.path.abspath(os.path.join(PROJECTS_ROOT, st.session_state.active_project, "outputs", "poses"))
            desired_video_dir = os.path.abspath(os.path.join(PROJECTS_ROOT, st.session_state.active_project, "outputs", "videos"))

            pose_dir = cfg.get("Pose", "pose_dir", fallback="").strip() if cfg.has_section("Pose") else ""
            video_dir = cfg.get("Videos", "video_dir", fallback="").strip() if cfg.has_section("Videos") else ""

            st.caption(f"pose_dir: {pose_dir or '(not set)'}")
            st.caption(f"video_dir: {video_dir or '(not set)'}")

            # Status checks
            if project_path:
                bp_path = os.path.join(project_path, "logs", "measures", "pose_configs", "bp_names", "project_bp_names.csv")
                vi_path = os.path.join(project_path, "logs", "video_info.csv")
                st.caption(f"project_bp_names.csv: {'OK' if os.path.exists(bp_path) else 'Missing'}")
                st.caption(f"video_info.csv: {'OK' if os.path.exists(vi_path) else 'Missing'}")

                if cfg_path and st.button("Generate bp_names + video_info now"):
                    from pipeline.simba_training import _ensure_project_bp_names, _ensure_video_info_csv, _read_project_path
                    proj = _read_project_path(cfg_path)
                    _ensure_project_bp_names(cfg_path, proj)
                    _ensure_video_info_csv(cfg_path, proj, v_stem, video_path=video_path)
                    st.success("Generated missing SimBA metadata files.")

                if cfg_path and st.button("Apply pose_dir/video_dir → outputs"):
                    if "Pose" not in cfg:
                        cfg["Pose"] = {}
                    if "Videos" not in cfg:
                        cfg["Videos"] = {}
                    cfg["Pose"]["pose_dir"] = desired_pose_dir
                    cfg["Videos"]["video_dir"] = desired_video_dir
                    with open(cfg_path, "w", encoding="utf-8") as f:
                        cfg.write(f)
                    st.success("Updated pose_dir and video_dir in SimBA config.")

                if cfg_path and st.button("Generate SimBA features_extracted now"):
                    try:
                        pose_setting = _ensure_simba_pose_setting(cfg_path)
                        st.caption(f"pose_estimation_body_parts: {pose_setting}")
                        result = _run_simba_feature_extraction(cfg_path, conda_env="simba_fast")
                        if result.returncode == 0:
                            st.success("SimBA features_extracted generated.")
                        else:
                            msg = (result.stderr or result.stdout or "").strip()
                            st.error(f"SimBA feature extraction failed: {msg or 'Unknown error'}")
                    except Exception as e:
                        st.error(f"SimBA feature extraction failed: {e}")
        except Exception as e:
            st.warning(f"Preflight check failed: {e}")

        """allow_fallback = st.checkbox(
            "Allow fallback features_extracted from train_merged (not SimBA format)",
            value=False,
            key="simba_allow_feature_fallback",
        )"""

        ann_dir = _get_project_dirs().get("annotations", "outputs/annotations")
        ann_files = sorted([p for p in Path(ann_dir).glob("*_behavior_annotations.csv") if p.is_file()])
        ann_stems = [p.name.replace("_behavior_annotations.csv", "") for p in ann_files]
        if ann_files:
            st.caption(f"Annotated videos found: {len(ann_files)}")
            st.caption(", ".join(ann_stems[:10]) + (" ..." if len(ann_stems) > 10 else ""))
        else:
            st.caption("Annotated videos found: 0")

        if st.button("🧪 Train SimBA Models (all annotated videos)", type="secondary"):
            try:
                if not st.session_state.get("active_project"):
                    st.error("No active project.")
                else:
                    prof = load_profile(PROJECTS_ROOT, st.session_state.active_project)
                    cfg_path = prof.get("simba", {}).get("config_path", "")
                    if not cfg_path or not os.path.exists(cfg_path):
                        st.error("SimBA config not found. Set it in Project Setup.")
                    elif not ann_files:
                        st.error("No annotation files found to train on.")
                    else:
                        labels = get_simba_labels(PROJECTS_ROOT, st.session_state.active_project)
                        if not labels:
                            st.error("No project labels set. Save labels in Project Setup.")
                        else:
                            merged_fallback = None
                            if st.session_state.get("rf_out_dir"):
                                cand_updated = os.path.join(st.session_state.rf_out_dir, "train_merged_updated.csv")
                                cand_base = os.path.join(st.session_state.rf_out_dir, "train_merged.csv")
                                if os.path.exists(cand_updated):
                                    merged_fallback = cand_updated
                                elif os.path.exists(cand_base):
                                    merged_fallback = cand_base
                            result = _run_simba_train_models_multi(
                                config_path=cfg_path,
                                annotations_csvs=[str(p) for p in ann_files],
                                video_stems=ann_stems,
                                labels=labels,
                                merged_csv_fallback=merged_fallback,
                                threshold=0.5,
                                min_bout=5,
                                conda_env="simba_fast",
                            )
                            if result.returncode == 0:
                                saved = []
                                if result.stdout:
                                    try:
                                        payload = json.loads(result.stdout.strip().splitlines()[-1])
                                        saved = payload.get("saved") or []
                                    except Exception:
                                        pass
                                st.success(f"Trained {len(saved)} SimBA models across {len(ann_files)} videos.")
                            else:
                                msg = (result.stderr or result.stdout or "").strip()
                                st.error(f"SimBA training failed: {msg or 'Unknown error'}")
            except Exception as e:
                st.error(f"SimBA training failed: {e}")

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
                            merged_fallback = None
                        if st.session_state.get("rf_out_dir"):
                            cand_updated = os.path.join(st.session_state.rf_out_dir, "train_merged_updated.csv")
                            cand_base = os.path.join(st.session_state.rf_out_dir, "train_merged.csv")
                            if os.path.exists(cand_updated):
                                merged_fallback = cand_updated
                            elif os.path.exists(cand_base):
                                merged_fallback = cand_base
                        result = _run_simba_train_models(
                            config_path=cfg_path,
                            annotations_csv=ann_path,
                            labels=labels,
                            video_stem=v_stem,
                            video_path=video_path,
                            merged_csv_fallback=merged_fallback,
                            threshold=0.5,
                            min_bout=5,
                            conda_env="simba_fast",
                        )
                        if result.returncode == 0:
                            saved = []
                            if result.stdout:
                                try:
                                    payload = json.loads(result.stdout.strip().splitlines()[-1])
                                    saved = payload.get("saved") or []
                                except Exception:
                                    pass
                            st.success(f"Trained {len(saved)} SimBA models.")
                        else:
                            msg = (result.stderr or result.stdout or "").strip()
                            st.error(f"SimBA training failed: {msg or 'Unknown error'}")
            except Exception as e:
                st.error(f"SimBA training failed: {e}")
