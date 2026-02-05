import os
import hashlib
from pathlib import Path
import streamlit as st
import glob

# 가상의 라이브러리 (사용자 환경에 맞춰 임포트)
from projects.projects import (
    list_projects, create_project, load_profile, set_dlc_config, import_dlc_config_file
)
from pipeline.run_pipeline import run_full_pipeline
from pipeline.preprocessing.crop import select_crop_roi, apply_crop
from pipeline.preprocessing.downsmaple import apply_downsample
from pipeline.preprocessing.trim import apply_trim
from pipeline.preprocessing.clahe import apply_clahe_to_video
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

def _delete_files(paths: list[str]) -> None:
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

    if st.sidebar.button("Open Project Setup", use_container_width=True, type="primary"):
        st.session_state.page = "project"
        st.rerun()

    st.sidebar.markdown("---")
    active = st.session_state.active_project or "(none)"
    if st.session_state.active_project and not st.session_state.active_profile:
        st.session_state.active_profile = load_profile(PROJECTS_ROOT, st.session_state.active_project)
    cfg = ""
    if st.session_state.active_profile:
        cfg = st.session_state.active_profile.get("dlc", {}).get("config_path", "")
    st.sidebar.caption(f"Active Project: {active}")
    st.sidebar.caption(f"Active Config: {cfg or '(none)'}")
    
    

    if st.session_state.active_project and st.session_state.active_project != "(no projects yet)":
        
        # 네비게이션
        st.sidebar.markdown("---")
        if st.sidebar.button("🏠 Home", use_container_width=True, type="primary"):
            st.session_state.page = "main"; st.rerun()
        

        st.sidebar.markdown("---")
        st.sidebar.header("📍 ROI Analysis")
        if st.sidebar.button("🎯 Draw ROIs"):
            st.session_state.page = "roi"
            st.rerun()

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
                    st.rerun()

        st.sidebar.markdown("---")
        st.sidebar.header("🧠 Labeling")
        if st.sidebar.button("📝 Behavior Annotator"):
            st.session_state.page = "annotate"
            st.rerun()

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
with st.container(border=True):
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
            if st.button("✂ Select ROI", use_container_width=True):
                roi = select_crop_roi(v_path)
                if roi: st.session_state.crop_roi = roi
            if st.session_state.crop_roi and st.button("Apply Crop", type="primary"):
                out = os.path.abspath(f"temp/crop_{os.path.basename(v_path)}")
                apply_crop(v_path, out, st.session_state.crop_roi)
                st.session_state.input_video_path = out; st.rerun()
        
        with c2:
            st.info("Step 2: Trim")
            t_cols = st.columns(2)
            ts = t_cols[0].text_input("Start(s)", "0")
            te = t_cols[1].text_input("End(s)", "10")
            if st.button("Cut Video", use_container_width=True):
                out = os.path.abspath(f"temp/trim_{os.path.basename(v_path)}")
                apply_trim(v_path, out, ts, te)
                st.session_state.input_video_path = out; st.rerun()

        with c3:
            st.info("Step 3: Resize")
            r_cols = st.columns(2)
            tw = r_cols[0].number_input("W", 1280, step=2)
            th = r_cols[1].number_input("H", 720, step=2)
            if st.button("Downsample", use_container_width=True):
                out = os.path.abspath(f"temp/down_{os.path.basename(v_path)}")
                apply_downsample(v_path, out, tw, th)
                st.session_state.input_video_path, st.session_state.resize_to = out, (tw, th)
                st.rerun()

        with c4:
            st.info("Step 4: Enhance")
            cl = st.slider("Contrast", 1.0, 5.0, 2.0)
            if st.button("Apply CLAHE", use_container_width=True):
                out = os.path.abspath(f"temp/en_{os.path.basename(v_path)}")
                apply_clahe_to_video(v_path, out, cl)
                st.session_state.input_video_path = out; st.rerun()

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
            
            if st.button("🔧 DLC: Extract Outliers", use_container_width=True):
                if conf_path:
                    import deeplabcut as dlc
                    dlc.extract_outlier_frames(config=conf_path, videos=[v_path])
                    st.success("Outliers extracted. Check DLC GUI.")
                else: st.error("Set DLC Config first.")

        if st.button("🚀 RUN FULL PIPELINE", type="primary", use_container_width=True):
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

                res = run_full_pipeline(
                    v_path, [], dlc_config_path=conf_path, out_dir=proj_out_dir,
                    force_pose=f_pose, force_analysis=f_anal,
                    roi=st.session_state.crop_roi, resize_to=st.session_state.resize_to,
                    output_name=st.session_state.original_video_name
                )
                
                # 결과 업데이트
                raw_out = res.get("out_video") or res.get("output_video")
                if raw_out:
                    p_path = os.path.abspath("temp/playable_result.mp4")
                    st.session_state.output_video = make_streamlit_playable_mp4(raw_out, p_path)
                
                st.session_state.kin_csv_path = res.get("kin_csv") or res.get("kinematics_csv")
                st.session_state.logs = res.get("logs", [])
                for k in ["speed_plot", "trajectory_plot", "trajectory_behavior", "turning_rate_plot_path", "trajectory_turning_plot", "nop_plot"]:
                    st.session_state[k] = res.get(k)
                st.rerun()

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
            st.dataframe(st.session_state.roi_result_df, use_container_width=True)

        # 탭을 이용한 플롯 분류
        tab1, tab2 = st.tabs(["📈 Basic Kinematics", "🎭 Behavior & NOP"])
        
        with tab1:
            tc1, tc2 = st.columns(2)
            if st.session_state.speed_plot and os.path.exists(st.session_state.speed_plot): 
                tc1.image(st.session_state.speed_plot, caption="Speed Over Time")
            if st.session_state.trajectory_plot and os.path.exists(st.session_state.trajectory_plot):
                tc2.image(st.session_state.trajectory_plot, caption="Full Trajectory")
        
        with tab2:
            tc1, tc2 = st.columns(2)
            if st.session_state.trajectory_behavior:
                tc1.image(st.session_state.trajectory_behavior, caption="Behavioral Map")
            if st.session_state.turning_rate_plot_path:
                tc2.image(st.session_state.turning_rate_plot_path, caption="Turning rate Map")
            #if st.session_state.nop_plot:
                #tc2.image(st.session_state.nop_plot, caption="NOP Analysis")

        with st.expander("📄 Detailed System Logs"):
            st.code("\n".join(st.session_state.logs))
else:
    st.info("Please upload a rodent video to begin.")