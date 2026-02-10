import os
import hashlib
from pathlib import Path
import streamlit as st
import glob
import pandas as pd
from typing import List

# 가상의 라이브러리 (사용자 환경에 맞춰 임포트)
from projects.projects import (
    list_projects, create_project, load_profile, set_dlc_config, import_dlc_config_file, set_dlc_reference
)
from pipeline.run_pipeline import run_full_pipeline
from pipeline.simba_backend import run_simba_pipeline
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

# --- 전역 설정 ---
PROJECTS_ROOT = "projects"
st.set_page_config(page_title="Rodent Kinematics Analyzer", layout="wide", page_icon="🐭")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.makedirs("temp", exist_ok=True)

init_session_state()

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
                    res_df, plot_p = run_multi_roi_analysis(
                        kin_csv=st.session_state.kin_csv_path,
                        roi_list=st.session_state.roi_list,
                        fps=30,
                        out_dir="outputs/roi",
                    )
                    st.session_state.roi_result_df = res_df
                    st.session_state.roi_result_plot = plot_p
                    _rerun()

        st.sidebar.markdown("---")
        st.sidebar.header("🧠 Labeling")
        if st.sidebar.button("📝 Behavior Annotator"):
            st.session_state.annotator_video_path = st.session_state.input_video_path
            st.session_state.clip_start = 0
            st.session_state.page = "annotate"
            _rerun()

render_sidebar()

if st.session_state.active_project:
    sync_project_data()
    
# --- [3] 페이지 라우팅 ---
if st.session_state.page == "project":
    render_project_setup_page(); st.stop()
elif st.session_state.page == "roi":
    render_roi_editor(); st.stop()
elif st.session_state.page == "annotate":
    if not st.session_state.input_video_path:
        st.warning("Please upload a video first."); st.stop()
    render_behavior_annotator_page(st.session_state.input_video_path); st.stop()

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
            tw = st.number_input("W", 1280, step=2)
            th = st.number_input("H", 720, step=2)
            if st.button("Downsample"):
                out = os.path.abspath(f"temp/down_{os.path.basename(v_path)}")
                apply_downsample(v_path, out, tw, th)
                st.session_state.input_video_path = out
                try:
                    cur_w, cur_h = get_video_size(v_path)
                except Exception:
                    cur_w, cur_h = None, None
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
            prefix = Path(st.session_state.original_video_name or v_path).stem
            
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
                        prefix = Path(st.session_state.original_video_name or v_path).stem
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
                            interpolation_setting=interp_opt,
                            smoothing_setting=smooth_opt,
                            smoothing_time_ms=smooth_time,
                            px_per_mm=px_per_mm if px_per_mm > 0 else None,
                            render_video=True,
                        )
                        st.session_state.logs = logs
                        st.session_state.simba_machine_csv = simba_results.get("simba_machine_csv")
                        st.session_state.simba_overlay_video = simba_results.get("simba_overlay_video")
                        _rerun()
        
        if st.button("🚀 RUN FULL PIPELINE"):
            if not conf_path: st.error("DLC Config path is missing!"); st.stop()
            with st.spinner("Processing... This may take a few minutes."):
                active = st.session_state.active_project
                proj_out_dir = os.path.join(PROJECTS_ROOT, active, "outputs")
                os.makedirs(proj_out_dir, exist_ok=True)
                prefix = Path(st.session_state.original_video_name or v_path).stem
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
                        },
                        roi=st.session_state.crop_roi, resize_to=st.session_state.resize_to,
                        output_name=st.session_state.original_video_name,
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
                st.session_state.logs = res.get("logs", [])
                st.session_state.simba_machine_csv = res.get("simba_machine_csv")
                st.session_state.simba_overlay_video = res.get("simba_overlay_video")
                
                for k in ["speed_plot", "trajectory_plot", "trajectory_behavior", "turning_rate_plot_path", "trajectory_turning_plot", "nop_plot"]:
                    st.session_state[k] = res.get(k)
                _rerun()

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

        # 탭 대신 섹션으로 렌더링 (구버전 Streamlit 호환)
        st.subheader("📈 Basic Kinematics")
        tc1, tc2 = st.columns(2)
        if st.session_state.speed_plot and os.path.exists(st.session_state.speed_plot): 
            tc1.image(st.session_state.speed_plot, caption="Speed Over Time")
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
            simba_df = pd.read_csv(st.session_state.simba_machine_csv)
            prob_cols = [c for c in simba_df.columns if c.lower().startswith("probability_") or c.lower().startswith("prob_")]
            if prob_cols:
                st.line_chart(simba_df[prob_cols])
            st.dataframe(simba_df.head(500))

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
