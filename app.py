import os
import hashlib
import shutil
import configparser
from pathlib import Path
import streamlit as st
import streamlit.elements.image as st_image
from streamlit.elements.lib.image_utils import image_to_url as _st_image_to_url
from streamlit.elements.lib.layout_utils import LayoutConfig
import glob
import pandas as pd
from typing import List
import matplotlib.pyplot as plt

# 가상의 라이브러리 (사용자 환경에 맞춰 임포트)
from projects.projects import (
    list_projects, create_project, load_profile, set_dlc_config, import_dlc_config_file, set_dlc_reference
)
from pipeline.run_pipeline import run_full_pipeline
from pipeline.simba_backend import run_simba_pipeline
from pipeline.simba_uncertain_segments import extract_uncertain_segments
from pipeline.SHAP.simba_shap import compute_simba_shap
from pipeline.weak_label import build_windowed_annotations
from pipeline.pose_dlc import run_deeplabcut_pose, extract_outlier_frames
from pipeline.preprocessing.crop import select_crop_roi, apply_crop
from pipeline.preprocessing.downsmaple import apply_downsample
from pipeline.preprocessing.trim import apply_trim
from pipeline.preprocessing.clahe import apply_clahe_to_video
from pipeline.preprocessing.video_size import get_video_size
from pipeline.ROI.ROI_anlaysis import run_multi_roi_analysis
from pipeline.ROI.ROIEditor import render_roi_editor
from pipeline.behavior_annotation.annotator_ui import render_behavior_annotator_page
from output_video import make_streamlit_playable_mp4
from pipeline.projectManagement.projectSetup import render_project_setup_page
from pipeline.projectManagement.syncProjectData import sync_project_data
from pipeline.projectManagement.init_session import init_session_state, reset_analysis_state, _get_latest_file
from pipeline.cache_utils import cached_pose_path, cached_kin_path, cached_turn_path, cached_beh_path

# --- 전역 설정 ---
PROJECTS_ROOT = "projects"
st.set_page_config(page_title="Rodent Kinematics Analyzer", layout="wide", page_icon="🐭")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.makedirs("temp", exist_ok=True)

# Compatibility for streamlit_drawable_canvas with newer Streamlit versions
if not hasattr(st_image, "image_to_url"):
    def image_to_url(image, width, clamp, channels, output_format, image_id):
        return _st_image_to_url(
            image,
            LayoutConfig(width=width),
            clamp,
            channels,
            output_format,
            image_id,
        )
    st_image.image_to_url = image_to_url

init_session_state()


def _running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

def _get_preprocess_signature(video_path: str | None = None):
    resize_to = st.session_state.resize_to
    v_path = video_path or st.session_state.input_video_path
    if resize_to and v_path:
        try:
            cur_size = get_video_size(v_path)
        except Exception:
            cur_size = None
        if cur_size:
            try:
                rt = tuple(resize_to)
            except Exception:
                rt = resize_to
            if rt == tuple(cur_size):
                resize_to = None
    return {
        "crop_roi": st.session_state.crop_roi,
        "resize_to": resize_to,
        "clahe_clip": st.session_state.get("clahe_clip"),
    }

def _strip_known_suffixes(stem: str) -> str:
    for suf in ("_kinematics", "_behavior", "_turning_rate", "_poses", "_filtered", "_features", "_ml_features"):
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem

def _find_targets_inserted_csv(kin_csv_path: str | None, active_project: str | None) -> str | None:
    if not kin_csv_path or not active_project:
        return None
    stem = _strip_known_suffixes(Path(kin_csv_path).stem)
    targets_dir = Path("projects") / active_project / "simba" / "project_folder" / "csv" / "targets_inserted"
    if not targets_dir.exists():
        return None
    exact = targets_dir / f"{stem}.csv"
    if exact.exists():
        return str(exact)
    csvs = list(targets_dir.glob("*.csv"))
    if len(csvs) == 1:
        return str(csvs[0])
    return None

def _plot_simba_ethogram(simba_csv: str, out_path: str) -> str | None:
    try:
        df = pd.read_csv(simba_csv)
    except Exception:
        return None

    exclude_prefixes = (
        "movement_", "all_bp_", "sum_", "mean_", "low_prob_", "probability_", "prob_"
    )
    exclude_exact = {"frame", "frames", "unnamed: 0", "sum_probabilities", "mean_probabilities"}
    label_cols = []
    for c in df.columns:
        lc = str(c).lower()
        if lc in exclude_exact or lc.startswith(exclude_prefixes):
            continue
        col = df[c]
        if col.dropna().isin([0, 1]).all():
            label_cols.append(c)
    if not label_cols:
        return None

    if "frame" in df.columns:
        frames = df["frame"].to_numpy()
    elif "Frame" in df.columns:
        frames = df["Frame"].to_numpy()
    else:
        frames = df.index.to_numpy()

    fig, ax = plt.subplots(figsize=(14, max(3, len(label_cols) * 0.4)))
    cmap = plt.cm.get_cmap("tab20", len(label_cols))
    label_color = {label: cmap(i) for i, label in enumerate(label_cols)}
    for i, label in enumerate(label_cols, start=1):
        on = df[label].to_numpy().astype(bool)
        if not on.any():
            continue
        start = None
        for idx, v in enumerate(on):
            if v and start is None:
                start = idx
            elif (not v) and (start is not None):
                ax.hlines(i, frames[start], frames[idx - 1], linewidth=6, color=label_color[label])
                start = None
        if start is not None:
            ax.hlines(i, frames[start], frames[len(frames) - 1], linewidth=6, color=label_color[label])

    ax.set_yticks(range(1, len(label_cols) + 1))
    ax.set_yticklabels(label_cols)
    ax.set_xlabel("Frame")
    ax.set_title("SimBA Behavior Ethogram")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path

def _plot_simba_probabilities(simba_csv: str, out_path: str) -> str | None:
    try:
        df = pd.read_csv(simba_csv)
    except Exception:
        return None
    prob_cols = [c for c in df.columns if str(c).lower().startswith(("probability_", "prob_"))]
    if not prob_cols:
        return None
    fig, ax = plt.subplots(figsize=(14, 6))
    for c in prob_cols:
        ax.plot(df.index, df[c].astype(float), label=c.replace("Probability_", "").replace("probability_", ""))
    ax.set_xlabel("Frame")
    ax.set_ylabel("Probability")
    ax.set_title("SimBA Probabilities")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def _delete_files(paths: List[str]) -> None:
    for p in paths:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

def _clear_project_cache(proj_out_dir: str, prefix: str, *, clear_pose: bool, clear_analysis: bool) -> None:
    if clear_pose:
        _delete_files([
            os.path.join(proj_out_dir, "poses", f"{prefix}_filtered.csv"),
        ])
    if clear_analysis:
        patterns = [
            os.path.join(proj_out_dir, "kinematics", f"{prefix}_*.csv"),
            os.path.join(proj_out_dir, "behavior", f"{prefix}_*.csv"),
            os.path.join(proj_out_dir, "ml", f"{prefix}_*.csv"),
            os.path.join(proj_out_dir, "plots", f"{prefix}_*.png"),
            os.path.join(proj_out_dir, "nop", f"{prefix}_*.csv"),
        ]
        for pat in patterns:
            for p in glob.glob(pat):
                try:
                    os.remove(p)
                except Exception:
                    pass



# --- [3] Sidebar ---
def render_sidebar():
    st.sidebar.header("🧪 Project Management")
    projects = list_projects(PROJECTS_ROOT)

    if st.sidebar.button("Open Project Setup"):
        st.session_state.page = "project"
        _rerun()

    st.sidebar.markdown("---")
    active = st.session_state.active_project or "(none)"
    if st.session_state.active_project and not st.session_state.active_profile:
        st.session_state.active_profile = load_profile(PROJECTS_ROOT, st.session_state.active_project)
    cfg = ""
    if st.session_state.active_profile:
        cfg = st.session_state.active_profile.get("dlc", {}).get("config_path", "")
    simba_cfg = ""
    if st.session_state.active_profile:
        simba_cfg = st.session_state.active_profile.get("simba", {}).get("config_path", "")
    st.sidebar.caption(f"Active Project: {active}")
    st.sidebar.caption(f"Active Config: {cfg or '(none)'}")
    st.sidebar.caption(f"Active SimBA Config: {simba_cfg or '(none)'}")
    
    

    if st.session_state.active_project and st.session_state.active_project != "(no projects yet)":
        
        # 네비게이션
        st.sidebar.markdown("---")
        if st.sidebar.button("🏠 Home"):
            st.session_state.page = "main"; _rerun()
        

        st.sidebar.markdown("---")
        st.sidebar.header("📍 ROI Analysis")
        if st.sidebar.button("🎯 Draw ROIs"):
            st.session_state.page = "roi"
            _rerun()

        st.sidebar.write(f"Active ROIs: **{len(st.session_state.roi_list)}**")

        if st.sidebar.button("▶ Run ROI Analysis"):
            if not st.session_state.kin_csv_path:
                st.sidebar.error("Run 'Analyze Video' first.")
            elif not st.session_state.roi_list:
                st.sidebar.error("No ROIs defined.")
            else:
                with st.spinner("Analyzing ROIs..."):
                    targets_csv = _find_targets_inserted_csv(
                        st.session_state.kin_csv_path, st.session_state.active_project
                    )
                    beh_csv = targets_csv or st.session_state.simba_machine_csv or st.session_state.beh_csv_path
                    res_df, plot_p = run_multi_roi_analysis(
                        kin_csv=st.session_state.kin_csv_path,
                        roi_list=st.session_state.roi_list,
                        fps=30,
                        out_dir="outputs/roi",
                        beh_csv=beh_csv,
                    )
                    st.session_state.roi_result_df = res_df
                    st.session_state.roi_result_plot = plot_p
                    roi_visits_plot = os.path.join("outputs", "roi", "multi_roi_visits_plot.png")
                    st.session_state.roi_visits_plot = roi_visits_plot if os.path.exists(roi_visits_plot) else None
                    _rerun()

        st.sidebar.markdown("---")
        st.sidebar.header("Labeling")
        if st.sidebar.button("Labeling: Initial"):
            st.session_state.annotator_video_path = st.session_state.input_video_path
            st.session_state.clip_start = 0
            st.session_state.page = "labeling_initial"
            _rerun()
        if st.sidebar.button("Labeling: Outlier"):
            st.session_state.annotator_video_path = st.session_state.input_video_path
            st.session_state.clip_start = 0
            st.session_state.page = "labeling_outlier"
            _rerun()
        if st.sidebar.button("Labeling: SimBA"):
            st.session_state.annotator_video_path = st.session_state.input_video_path
            st.session_state.clip_start = 0
            st.session_state.page = "labeling_simba"
            _rerun()

if _running_in_streamlit():
    render_sidebar()

    if st.session_state.active_project:
        sync_project_data()
    
    # --- [3] 페이지 라우팅 ---
    if st.session_state.page == "project":
        render_project_setup_page(); st.stop()
    elif st.session_state.page == "roi":
        render_roi_editor(); st.stop()
    elif st.session_state.page == "labeling_initial":
        if not st.session_state.input_video_path:
            st.warning("Please upload a video first."); st.stop()
        render_behavior_annotator_page(st.session_state.input_video_path, page_mode="labeling"); st.stop()
    elif st.session_state.page == "labeling_outlier":
        if not st.session_state.input_video_path:
            st.warning("Please upload a video first."); st.stop()
        render_behavior_annotator_page(st.session_state.input_video_path, page_mode="outlier"); st.stop()
    elif st.session_state.page == "labeling_simba":
        if not st.session_state.input_video_path:
            st.warning("Please upload a video first."); st.stop()
        render_behavior_annotator_page(st.session_state.input_video_path, page_mode="simba"); st.stop()

    # --- [4] 메인 화면 ---
    st.title("Rodent Kinematics Analyzer 🐭")

    # 비디오 업로드 영역
    with st.container():
        uploaded = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov", "mkv"])
        if uploaded:
            fid = hashlib.md5(uploaded.getbuffer()).hexdigest()
            if st.session_state.uploaded_file_id != fid:
                reset_analysis_state()
                st.session_state.uploaded_file_id = fid
                st.session_state.original_video_name = uploaded.name
                t_path = os.path.join("temp", uploaded.name)
                with open(t_path, "wb") as f: f.write(uploaded.getbuffer())
                st.session_state.input_video_path = t_path

    if st.session_state.input_video_path:
        v_path = st.session_state.input_video_path
    
        # 1. 전처리 도구
        with st.expander("🛠️ Video Preprocessing Step", expanded=not st.session_state.kin_csv_path):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.info("Step 1: Crop")
                if st.button("✂ Select ROI"):
                    roi = select_crop_roi(v_path)
                    if roi: st.session_state.crop_roi = roi
                if st.session_state.crop_roi and st.button("Apply Crop"):
                    out = os.path.abspath(f"temp/crop_{os.path.basename(v_path)}")
                    apply_crop(v_path, out, st.session_state.crop_roi)
                    st.session_state.input_video_path = out; _rerun()
        
            with c2:
                st.info("Step 2: Trim")
                ts = st.text_input("Start(s)", "0")
                te = st.text_input("End(s)", "10")
                if st.button("Cut Video"):
                    out = os.path.abspath(f"temp/trim_{os.path.basename(v_path)}")
                    apply_trim(v_path, out, ts, te)
                    st.session_state.input_video_path = out; _rerun()

            with c3:
                st.info("Step 3: Resize")
                try:
                    cur_w, cur_h = get_video_size(v_path)
                except Exception:
                    cur_w, cur_h = None, None
                if cur_w and cur_h:
                    aspect = cur_h / cur_w
                else:
                    aspect = 720 / 1280

                lock_ar = st.checkbox("Lock aspect ratio", value=True, key="resize_lock_ar")
                tw = st.number_input(
                    "W",
                    min_value=320,
                    value=int(cur_w) if cur_w else 1280,
                    step=2,
                    key="resize_w",
                )
                if lock_ar:
                    th = int(round((tw * aspect) / 2) * 2)
                    th = max(2, th)
                    st.caption(f"H (auto): {th}")
                else:
                    th = st.number_input(
                        "H",
                        min_value=180,
                        value=int(cur_h) if cur_h else 720,
                        step=2,
                        key="resize_h",
                    )
                if st.button("Downsample"):
                    out = os.path.abspath(f"temp/down_{os.path.basename(v_path)}")
                    apply_downsample(v_path, out, tw, th)
                    st.session_state.input_video_path = out
                    if cur_w is not None and cur_h is not None and (tw, th) == (cur_w, cur_h):
                        st.session_state.resize_to = None
                    else:
                        st.session_state.resize_to = (tw, th)
                    _rerun()

            with c4:
                st.info("Step 4: Enhance")
                cl = st.slider("Contrast", 1.0, 5.0, 2.0)
                if st.button("Apply CLAHE"):
                    out = os.path.abspath(f"temp/en_{os.path.basename(v_path)}")
                    apply_clahe_to_video(v_path, out, cl)
                    st.session_state.input_video_path = out; _rerun()
                    st.session_state.input_video_path = out
                    st.session_state.clahe_clip = cl
                    _rerun()


        # 2. 분석 실행 및 프리뷰
        col_pre, col_res = st.columns(2)
        with col_pre:
            st.subheader("📺 Input Preview")
            st.video(v_path)
            st.caption(f"Current Path: {v_path}")
        
            with st.expander("⚙️ Pipeline Options"):
                f_pose = st.checkbox("Force Re-run Pose Estimation")
                f_anal = st.checkbox("Force Re-run Kinematics")
                conf_path = st.session_state.active_profile.get("dlc", {}).get("config_path") if st.session_state.active_profile else None
                simba_conf_path = st.session_state.active_profile.get("simba", {}).get("config_path") if st.session_state.active_profile else None
                prefix = Path(v_path).stem
            
                if st.button("🔧 DLC: Extract Outliers"):
                    if conf_path:
                        try:
                            extract_outlier_frames(
                                config_path=conf_path,
                                video_path=v_path,
                                logs=st.session_state.logs,
                            )
                            st.success("Outliers extracted. Check DLC GUI.")
                        except Exception as e:
                            st.error(f"Extract outliers failed: {e}")                    
                            st.success("Outliers extracted. Check DLC GUI.")
                    else: st.error("Set DLC Config first.")

            with st.expander("SimBA Options"):
                st.subheader("SimBA Options")
                run_simba = st.checkbox("Run SimBA (full pipeline)", value=False)
                px_per_mm = st.number_input("SimBA px/mm (optional)", min_value=0.0, step=0.1, value=0.0)
                augment_kin = st.checkbox("Augment features with kinematics (requires retrained models)", value=False)
                interp_opt = st.selectbox(
                    "Interpolation",
                    ["None", "Body-parts: Nearest", "Body-parts: Linear", "Body-parts: Quadratic",
                "Animal(s): Nearest", "Animal(s): Linear", "Animal(s): Quadratic"],
                    index=0,
                )
                smooth_opt = st.selectbox(
                    "Smoothing",
                    ["None", "Savitzky-Golay", "Gaussian"],
                    index=0,
                )
                smooth_time = st.number_input("Smoothing time (ms)", min_value=0, step=50, value=0)
                st.markdown("---")
                st.caption("Uncertain segments (for active learning)")
                max_prob_low = st.number_input("Max prob low", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
                max_prob_high = st.number_input("Max prob high", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
                gap_thresh = st.number_input("Top-2 prob gap < ", min_value=0.0, max_value=1.0, value=0.15, step=0.05)
                min_dur = st.number_input("Min duration (sec)", min_value=0.0, value=0.5, step=0.1)
                if st.button("Generate Uncertain Segments CSV"):
                    try:
                        machine_csv = st.session_state.simba_machine_csv
                        if not machine_csv or not os.path.exists(machine_csv):
                            st.error("SimBA machine_results not found. Run SimBA inference first.")
                        else:
                            out_csv = extract_uncertain_segments(
                                machine_csv,
                                max_prob_low=max_prob_low,
                                max_prob_high=max_prob_high,
                                gap_thresh=gap_thresh,
                                min_duration_sec=min_dur,
                            )
                            st.success(f"Saved: {out_csv}")
                            try:
                                seg_df = pd.read_csv(out_csv)
                                st.dataframe(seg_df.head(200))
                            except Exception:
                                pass
                    except Exception as e:
                        st.error(f"Failed to generate segments: {e}")

                if st.button("Create SimBA _filtered features_extracted"):
                    if not simba_conf_path:
                        st.error("SimBA config path is missing!")
                    else:
                        try:
                            cfg = configparser.ConfigParser()
                            cfg.read(simba_conf_path, encoding="utf-8")
                            project_dir = ""
                            for section in ("General settings", "Project"):
                                if cfg.has_section(section) and cfg.has_option(section, "project_path"):
                                    project_dir = cfg.get(section, "project_path", fallback="").strip()
                                    if project_dir:
                                        break
                            if not project_dir:
                                raise RuntimeError("Could not resolve SimBA project_path from config.")

                            feats_dir = os.path.join(project_dir, "csv", "features_extracted")
                            os.makedirs(feats_dir, exist_ok=True)
                            stem = Path(v_path).stem
                            src = os.path.join(feats_dir, f"{stem}.csv")
                            dst = os.path.join(feats_dir, f"{stem}_filtered.csv")
                            if not os.path.exists(src):
                                st.error(f"Source features_extracted not found: {src}")
                            else:
                                shutil.copy(src, dst)
                                st.success(f"Created: {dst}")
                        except Exception as e:
                            st.error(f"Create filtered features failed: {e}")

                if run_simba and not simba_conf_path:
                    st.warning("SimBA config path is missing. Set it in Project Setup.")

                if st.button("🧪 Run SimBA Prediction Only"):
                    if not simba_conf_path:
                        st.error("SimBA config path is missing!")
                    elif not conf_path:
                        st.error("DLC Config path is missing!")
                    else:
                        with st.spinner("Running SimBA inference..."):
                            logs = []
                            active = st.session_state.active_project
                            proj_out_dir = os.path.join(PROJECTS_ROOT, active, "outputs")
                            os.makedirs(proj_out_dir, exist_ok=True)
                            prefix = Path(v_path).stem
                            st.subheader("DLC Live Logs")
                            dlc_log_box = st.empty()
                            dlc_lines = []
                            def _dlc_log(line: str) -> None:
                                dlc_lines.append(line)
                                if len(dlc_lines) > 200:
                                    del dlc_lines[:-200]
                                dlc_log_box.code("\n".join(dlc_lines))
                            try:
                                pose_csv = run_deeplabcut_pose(
                                    v_path,
                                    logs,
                                    CONFIG_PATH=conf_path,
                                    force=f_pose,
                                    out_dir=proj_out_dir,
                                    cache_key=prefix,
                                    log_callback=_dlc_log,
                                )
                                st.session_state.pose_csv_path = pose_csv
                                st.session_state.pipeline_input_video = v_path
                            except Exception as e:
                                st.session_state.logs = logs
                                st.error(f"DLC failed: {e}")
                                st.stop()
                            
                            simba_results = run_simba_pipeline(
                                config_path=simba_conf_path,
                                pose_csv=pose_csv,
                                input_video=v_path,
                                logs=logs,
                                out_dir=proj_out_dir,
                                force=f_anal,
                                kin_csv=st.session_state.kin_csv_path if st.session_state.kin_csv_path else None,
                                augment_with_kinematics=augment_kin,
                                interpolation_setting=interp_opt,
                                smoothing_setting=smooth_opt,
                                smoothing_time_ms=smooth_time,
                                px_per_mm=px_per_mm if px_per_mm > 0 else None,
                                render_video=True,
                            )
                            st.session_state.logs = logs
                            st.session_state.simba_machine_csv = simba_results.get("simba_machine_csv")
                            st.session_state.simba_overlay_video = simba_results.get("simba_overlay_video")
                            if st.session_state.simba_machine_csv:
                                eth_path = os.path.join(PROJECTS_ROOT, active, "outputs", "simba", "simba_ethogram.png")
                                os.makedirs(os.path.dirname(eth_path), exist_ok=True)
                                st.session_state.simba_ethogram = _plot_simba_ethogram(
                                    st.session_state.simba_machine_csv, eth_path
                                )
                                prob_path = os.path.join(PROJECTS_ROOT, active, "outputs", "plots", "simba_probabilities.png")
                                os.makedirs(os.path.dirname(prob_path), exist_ok=True)
                                _plot_simba_probabilities(st.session_state.simba_machine_csv, prob_path)
        
            with st.expander("Pipeline Stages (Manual)"):
                st.caption("Run stages individually and reuse cached results. Avoids full pipeline reruns.")
                use_cached_input = st.checkbox("Reuse prepared input if available", value=True)
                stage_pose = st.checkbox("Stage 1: Pose (DLC)", value=False)
                stage_analysis = st.checkbox("Stage 2: Analysis (Kinematics/Behavior/Turning/NOP)", value=False)
                stage_render = st.checkbox("Stage 3: Render + ML features", value=False)
                stage_plots = st.checkbox("Stage 4: Plots", value=False)
                stage_simba = st.checkbox("Stage 5: SimBA", value=False)

                if not st.session_state.pose_csv_path or not st.session_state.kin_csv_path or not st.session_state.turn_csv_path or not st.session_state.beh_csv_path:
                    active = st.session_state.active_project
                    proj_out_dir = os.path.join(PROJECTS_ROOT, active, "outputs") if active else None
                    candidate_video = st.session_state.pipeline_input_video or v_path
                    if proj_out_dir and candidate_video:
                        cache_key = Path(candidate_video).stem
                        if not st.session_state.pose_csv_path:
                            pose_guess = cached_pose_path(candidate_video, proj_out_dir, cache_key)
                            if os.path.exists(pose_guess):
                                st.session_state.pose_csv_path = pose_guess
                        if not st.session_state.kin_csv_path:
                            kin_guess = cached_kin_path(candidate_video, proj_out_dir, cache_key)
                            if os.path.exists(kin_guess):
                                st.session_state.kin_csv_path = kin_guess
                        if not st.session_state.turn_csv_path:
                            turn_guess = cached_turn_path(candidate_video, proj_out_dir, cache_key)
                            if os.path.exists(turn_guess):
                                st.session_state.turn_csv_path = turn_guess
                        if not st.session_state.beh_csv_path:
                            beh_guess = cached_beh_path(candidate_video, proj_out_dir, cache_key)
                            if os.path.exists(beh_guess):
                                st.session_state.beh_csv_path = beh_guess
                    if proj_out_dir:
                        if not st.session_state.pose_csv_path:
                            st.session_state.pose_csv_path = _get_latest_file([str(Path(proj_out_dir) / "poses" / "*_filtered.csv")])
                        if not st.session_state.kin_csv_path:
                            st.session_state.kin_csv_path = _get_latest_file([str(Path(proj_out_dir) / "kinematics" / "*_kinematics.csv")])
                        if not st.session_state.turn_csv_path:
                            st.session_state.turn_csv_path = _get_latest_file([str(Path(proj_out_dir) / "kinematics" / "*_turning_rate.csv")])
                        if not st.session_state.beh_csv_path:
                            st.session_state.beh_csv_path = _get_latest_file([str(Path(proj_out_dir) / "behavior" / "*_behavior.csv")])

                if st.button("Run Selected Stages"):
                    if not conf_path:
                        st.error("DLC Config path is missing!")
                        st.stop()

                    logs = []
                    active = st.session_state.active_project
                    proj_out_dir = os.path.join(PROJECTS_ROOT, active, "outputs")
                    os.makedirs(proj_out_dir, exist_ok=True)

                    steps = []
                    input_video_override = None
                    if not use_cached_input or not st.session_state.pipeline_input_video:
                        steps.append("prepare")
                    else:
                        input_video_override = st.session_state.pipeline_input_video

                    if stage_pose:
                        steps.append("pose")
                    if stage_analysis:
                        steps += ["kinematics", "behavior", "turning", "nop"]
                    if stage_render:
                        steps.append("render")
                    if stage_plots:
                        steps.append("plots")
                    if stage_simba:
                        steps.append("simba")

                    if not steps:
                        st.warning("Select at least one stage.")
                        st.stop()

                    if "pose" not in steps and not st.session_state.pose_csv_path:
                        st.error("Pose CSV not found. Run Stage 1 first.")
                        st.stop()
                    if "kinematics" not in steps and not st.session_state.kin_csv_path:
                        st.error("Kinematics CSV not found. Run Stage 2 first.")
                        st.stop()
                    if "turning" not in steps and ("plots" in steps) and not st.session_state.turn_csv_path:
                        st.error("Turning-rate CSV not found. Run Stage 2 first.")
                        st.stop()

                    st.subheader("DLC Live Logs")
                    dlc_log_box = st.empty()
                    dlc_lines = []
                    def _dlc_log(line: str) -> None:
                        dlc_lines.append(line)
                        if len(dlc_lines) > 200:
                            del dlc_lines[:-200]
                        dlc_log_box.code("\n".join(dlc_lines))

                    with st.spinner("Running selected stages..."):
                        res = run_full_pipeline(
                            v_path, logs,
                            dlc_config_path=conf_path, out_dir=proj_out_dir,
                            force_pose=f_pose, force_analysis=f_anal,
                            run_simba=stage_simba,
                            simba_config_path=simba_conf_path,
                            simba_options={
                                "interpolation_setting": interp_opt,
                                "smoothing_setting": smooth_opt,
                                "smoothing_time_ms": smooth_time,
                                "px_per_mm": px_per_mm if px_per_mm > 0 else None,
                                "force": f_anal,
                                "augment_with_kinematics": augment_kin,
                            },
                            roi=st.session_state.crop_roi, resize_to=st.session_state.resize_to,
                            output_name=os.path.basename(v_path),
                            dlc_log_callback=_dlc_log,
                            steps=steps,
                            input_video_override=input_video_override,
                            pose_csv=st.session_state.pose_csv_path,
                            kin_csv=st.session_state.kin_csv_path,
                            beh_csv=st.session_state.beh_csv_path,
                            turn_csv=st.session_state.turn_csv_path,
                            simba_machine_csv=st.session_state.simba_machine_csv,
                        )

                    st.session_state.logs = logs
                    if res.get("input_video"):
                        st.session_state.pipeline_input_video = res.get("input_video")
                    if res.get("pose_csv"):
                        st.session_state.pose_csv_path = res.get("pose_csv")
                    if res.get("kin_csv"):
                        st.session_state.kin_csv_path = res.get("kin_csv")
                    if res.get("beh_csv"):
                        st.session_state.beh_csv_path = res.get("beh_csv")
                    if res.get("turn_csv"):
                        st.session_state.turn_csv_path = res.get("turn_csv")

                    if res.get("out_video"):
                        p_path = os.path.abspath("temp/playable_result.mp4")
                        st.session_state.output_video = make_streamlit_playable_mp4(res.get("out_video"), p_path)

                    st.session_state.simba_machine_csv = res.get("simba_machine_csv") or st.session_state.simba_machine_csv
                    st.session_state.simba_overlay_video = res.get("simba_overlay_video") or st.session_state.simba_overlay_video
                    if st.session_state.simba_machine_csv:
                        eth_path = os.path.join(PROJECTS_ROOT, active, "outputs", "simba", "simba_ethogram.png")
                        os.makedirs(os.path.dirname(eth_path), exist_ok=True)
                        st.session_state.simba_ethogram = _plot_simba_ethogram(
                            st.session_state.simba_machine_csv, eth_path
                        )
                        prob_path = os.path.join(PROJECTS_ROOT, active, "outputs", "plots", "simba_probabilities.png")
                        os.makedirs(os.path.dirname(prob_path), exist_ok=True)
                        _plot_simba_probabilities(st.session_state.simba_machine_csv, prob_path)

                    for k in ["speed_plot", "trajectory_plot", "trajectory_behavior", "turning_rate_plot_path", "trajectory_turning_plot", "nop_plot"]:
                        if res.get(k) is not None:
                            st.session_state[k] = res.get(k)

            if st.button("🚀 RUN FULL PIPELINE"):
                if not conf_path: st.error("DLC Config path is missing!"); st.stop()
                with st.spinner("Processing... This may take a few minutes."):
                    active = st.session_state.active_project
                    proj_out_dir = os.path.join(PROJECTS_ROOT, active, "outputs")
                    os.makedirs(proj_out_dir, exist_ok=True)
                    prefix = Path(v_path).stem
                    if f_pose or f_anal:
                        _clear_project_cache(
                            proj_out_dir,
                            prefix,
                            clear_pose=f_pose,
                            clear_analysis=f_anal,
                        )

                    st.subheader("DLC Live Logs")
                    dlc_log_box = st.empty()
                    dlc_lines = []
                    def _dlc_log(line: str) -> None:
                        dlc_lines.append(line)
                        if len(dlc_lines) > 200:
                            del dlc_lines[:-200]
                        dlc_log_box.code("\n".join(dlc_lines))

                    try:
                        res = run_full_pipeline(
                            v_path, [], dlc_config_path=conf_path, out_dir=proj_out_dir,
                            force_pose=f_pose, force_analysis=f_anal,
                            run_simba=run_simba,
                            simba_config_path=simba_conf_path,
                            simba_options={
                                "interpolation_setting": interp_opt,
                                "smoothing_setting": smooth_opt,
                                "smoothing_time_ms": smooth_time,
                                "px_per_mm": px_per_mm if px_per_mm > 0 else None,
                                "force": f_anal,
                                "augment_with_kinematics": augment_kin,
                            },
                            roi=st.session_state.crop_roi, resize_to=st.session_state.resize_to,
                            output_name=os.path.basename(v_path),
                            dlc_log_callback=_dlc_log,
                        )
                    except Exception as e:
                        st.error(f"Pipeline failed: {e}")
                        st.stop()
                
                    # 결과 업데이트
                    raw_out = res.get("out_video") or res.get("output_video")
                    if raw_out:
                        p_path = os.path.abspath("temp/playable_result.mp4")
                        st.session_state.output_video = make_streamlit_playable_mp4(raw_out, p_path)
                
                    st.session_state.kin_csv_path = res.get("kin_csv") or res.get("kinematics_csv")
                    st.session_state.pipeline_input_video = res.get("input_video")
                    st.session_state.pose_csv_path = res.get("pose_csv")
                    st.session_state.beh_csv_path = res.get("beh_csv")
                    st.session_state.turn_csv_path = res.get("turn_csv")
                    st.session_state.logs = res.get("logs", [])
                    st.session_state.simba_machine_csv = res.get("simba_machine_csv")
                    st.session_state.simba_overlay_video = res.get("simba_overlay_video")
                    if st.session_state.simba_machine_csv:
                        eth_path = os.path.join(PROJECTS_ROOT, active, "outputs", "simba", "simba_ethogram.png")
                        os.makedirs(os.path.dirname(eth_path), exist_ok=True)
                        st.session_state.simba_ethogram = _plot_simba_ethogram(
                            st.session_state.simba_machine_csv, eth_path
                        )
                        prob_path = os.path.join(PROJECTS_ROOT, active, "outputs", "plots", "simba_probabilities.png")
                        os.makedirs(os.path.dirname(prob_path), exist_ok=True)
                        _plot_simba_probabilities(st.session_state.simba_machine_csv, prob_path)
                
                    for k in ["speed_plot", "trajectory_plot", "trajectory_behavior", "turning_rate_plot_path", "trajectory_turning_plot", "nop_plot"]:
                        st.session_state[k] = res.get(k)

        with col_res:
            st.subheader("🎬 Analysis Result")
            if st.session_state.output_video:
                st.video(st.session_state.output_video)
                with open(st.session_state.output_video, "rb") as f:
                    st.download_button("📥 Download Result Video", f, file_name=f"analyzed_{st.session_state.active_project}.mp4")
            else:
                st.info("Run the pipeline to see the annotated video.")

        # 3. 데이터 시각화 대시보드
        if st.session_state.kin_csv_path:
            sync_project_data()
            st.markdown("---")
            st.header("📊 Kinematics Dashboard")

            if st.session_state.roi_result_plot:
                st.subheader("ROI Analysis")
                st.image(st.session_state.roi_result_plot)
                st.dataframe(st.session_state.roi_result_df)
                if st.session_state.get("roi_visits_plot") and os.path.exists(st.session_state.roi_visits_plot):
                    st.subheader("ROI Visits Summary")
                    st.image(st.session_state.roi_visits_plot)

            # 탭 대신 섹션으로 렌더링 (구버전 Streamlit 호환)
            st.subheader("📈 Basic Kinematics")
            tc1, tc2 = st.columns(2)
            # Speed plot saved on disk only; not shown in Streamlit to avoid delays.
            if st.session_state.speed_plot and os.path.exists(st.session_state.speed_plot):
                tc1.image(st.session_state.speed_plot, caption="Speed Plot")
            if st.session_state.trajectory_plot and os.path.exists(st.session_state.trajectory_plot):
                tc2.image(st.session_state.trajectory_plot, caption="Full Trajectory")

            st.subheader("🎭 Behavior")
            tc1, tc2 = st.columns(2)
            if st.session_state.trajectory_behavior and os.path.exists(st.session_state.trajectory_behavior):
                tc1.image(st.session_state.trajectory_behavior, caption="Behavioral Map")
            if st.session_state.turning_rate_plot_path and os.path.exists(st.session_state.turning_rate_plot_path):
                tc2.image(st.session_state.turning_rate_plot_path, caption="Turning rate Map")
            #if st.session_state.nop_plot:
                #tc2.image(st.session_state.nop_plot, caption="NOP Analysis")

            st.subheader("🧪 SimBA")
            if st.session_state.simba_machine_csv and os.path.exists(st.session_state.simba_machine_csv):
                st.subheader("SimBA Predictions")
                st.dataframe(pd.read_csv(st.session_state.simba_machine_csv).head(500))
                eth_path = st.session_state.get("simba_ethogram")
                if not eth_path and st.session_state.active_project:
                    eth_path = os.path.join(PROJECTS_ROOT, st.session_state.active_project, "outputs", "simba", "simba_ethogram.png")
                    if os.path.exists(eth_path):
                        st.session_state.simba_ethogram = eth_path
                if eth_path and os.path.exists(eth_path):
                    st.subheader("SimBA Ethogram")
                    st.image(eth_path)

            with st.expander("SHAP (SimBA)", expanded=False):
                simba_conf_path = st.session_state.active_profile.get("simba", {}).get("config_path") if st.session_state.active_profile else None
                st.caption("Compute SHAP for SimBA models. This can take several minutes.")
                shap_labels = st.text_input("Labels (comma-separated)", value="")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    shap_present = st.number_input("Target-present samples", min_value=10, max_value=2000, value=100, step=10)
                with col_b:
                    shap_absent = st.number_input("Target-absent samples", min_value=10, max_value=2000, value=100, step=10)
                with col_c:
                    shap_video_stem = st.text_input("Video stem (optional)", value="")

                if st.button("Compute SHAP"):
                    if not simba_conf_path:
                        st.error("SimBA config path is missing. Set it in Project Setup.")
                    else:
                        with st.spinner("Computing SHAP..."):
                            labels = [x.strip() for x in shap_labels.split(",") if x.strip()] or None
                            try:
                                outputs = compute_simba_shap(
                                    simba_conf_path,
                                    labels=labels,
                                    video_stem=shap_video_stem or None,
                                    n_present=int(shap_present),
                                    n_absent=int(shap_absent),
                                )
                                st.success(f"SHAP complete. Outputs: {len(outputs)} files.")
                                shap_pngs = [p for p in outputs if p.lower().endswith(".png")]
                                for p in shap_pngs:
                                    if os.path.exists(p):
                                        st.image(p, caption=os.path.basename(p))
                            except Exception as e:
                                st.error(f"SHAP failed: {e}")

            with st.expander("Pseudo-labels (SimBA)", expanded=False):
                st.caption("Create annotation CSVs from weak labels.")
                default_machine = st.session_state.simba_machine_csv if st.session_state.simba_machine_csv else ""
                machine_csv = st.text_input("machine_results CSV path", value=default_machine)
                pseudo_labels = st.text_input(
                    "Labels (comma-separated)",
                    value="sniffing_up, sniffing_down, grooming, rearing, turning, moving, rest, running, top, other",
                )
                col_a, col_b = st.columns(2)
                with col_a:
                    pseudo_threshold = st.number_input("Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
                with col_b:
                    pseudo_min_bout = st.number_input("Min bout length", min_value=1, max_value=500, value=5, step=1)

                st.markdown("---")
                st.caption("Create windowed annotations from weak labels (labels every fixed window).")
                col_w1, col_w2, col_w3 = st.columns(3)
                with col_w1:
                    win_len = st.number_input("Window length (frames)", min_value=1, max_value=10000, value=30, step=1)
                with col_w2:
                    win_stride = st.number_input("Window stride (frames)", min_value=1, max_value=10000, value=15, step=1)
                with col_w3:
                    use_none = st.checkbox("Use 'none' below threshold", value=False)
                win_out_dir = st.text_input("Annotation output folder (optional)", value="outputs/annotations")
                win_thresh = st.number_input("Window threshold (optional)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
                if st.button("Create Annotation CSV (Windowed Weak Labels)"):
                    if not machine_csv or not os.path.exists(machine_csv):
                        st.error("machine_results CSV not found.")
                    else:
                        labels = [x.strip() for x in pseudo_labels.split(",") if x.strip()]
                        if not labels:
                            st.error("Please provide at least one label.")
                        else:
                            try:
                                stem = Path(machine_csv).stem
                                if stem.endswith("_filtered"):
                                    stem = stem[: -len("_filtered")]
                                out_dir = win_out_dir.strip() or "outputs/annotations"
                                os.makedirs(out_dir, exist_ok=True)
                                out_csv = os.path.join(out_dir, f"{stem}_behavior_annotations.csv")
                                thresh = float(win_thresh) if use_none else None
                                out_path = build_windowed_annotations(
                                    machine_csv=machine_csv,
                                    labels=labels,
                                    window_len=int(win_len),
                                    stride=int(win_stride),
                                    threshold=thresh,
                                    out_csv=out_csv,
                                )
                                st.success(f"Saved: {out_path}")
                            except Exception as e:
                                st.error(f"Windowed annotation CSV failed: {e}")

            if st.session_state.simba_overlay_video and os.path.exists(st.session_state.simba_overlay_video):
                st.subheader("SimBA Overlay Video")
                try:
                    simba_playable = make_streamlit_playable_mp4(
                        st.session_state.simba_overlay_video,
                        os.path.abspath("temp/playable_simba_overlay.mp4"),
                    )
                    st.video(simba_playable)
                except Exception:
                    st.video(st.session_state.simba_overlay_video)
            else:
                st.info("Run SimBA to see predictions and overlays.")

            with st.expander("📄 Detailed System Logs"):
                st.code("\n".join(st.session_state.logs))
    else:
        st.info("Please upload a rodent video to begin.")
