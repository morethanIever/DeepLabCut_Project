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

# --- 전역 설정 ---
PROJECTS_ROOT = "projects"
st.set_page_config(page_title="Rodent Kinematics Analyzer", layout="wide", page_icon="🐭")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.makedirs("temp", exist_ok=True)

# --- [1] Session State & Utils ---
def init_session_state():
    defaults = {
        "uploaded_file_id": None, "input_video_path": None, "output_video": None,
        "logs": [], "page": "main", "kin_csv_path": None, "crop_roi": None, "resize_to": None,
        "speed_plot": None, "trajectory_plot": None, "trajectory_behavior": None,
        "turning_rate_plot_path": None, "trajectory_turning_plot": None, "nop_plot": None,
        "roi_list": [], "roi_radius": 80, "roi_result_df": None, "roi_result_plot": None,
        "active_project": None, "active_profile": None,
        "original_video_name": None,
        "experiment_type": "Open Field", "camera_view": "Top View"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_analysis_state():
    keys = ["output_video", "logs", "speed_plot", "trajectory_plot", "trajectory_behavior",
            "turning_rate_plot_path", "trajectory_turning_plot", "nop_plot", "kin_csv_path",
            "roi_result_df", "roi_result_plot"]
    for k in keys: st.session_state[k] = None
    st.session_state.logs = []

def _get_latest_file(pattern_list):
    all_files = []
    for p in pattern_list:
        all_files.extend(glob.glob(str(p), recursive=True))
    valid_files = [f for f in all_files if os.path.exists(f)]
    return max(valid_files, key=os.path.getmtime) if valid_files else None

def sync_project_data():
    """프로젝트 폴더 내의 결과물을 CSV 해시 파일명에 맞춰 정밀하게 동기화"""
    active = st.session_state.active_project
    if not active or active == "(no projects yet)": return

    base_dir = Path(PROJECTS_ROOT) / active / "outputs"
    plots_dir = base_dir / "plots"
    if not plots_dir.exists(): return

    # 1. Kinematics CSV 찾기 (가장 최근 분석된 데이터 기준)
    if not st.session_state.kin_csv_path:
        st.session_state.kin_csv_path = _get_latest_file([base_dir / "**/*kin*.csv"])

    # 2. CSV가 있으면 해당 해시값(prefix)으로 시작하는 모든 PNG를 긁어옴
    if st.session_state.kin_csv_path:
        csv_stem = Path(st.session_state.kin_csv_path).stem
        prefix = csv_stem.split('_')[0] # '3442abf8...' 추출

        # 매핑 규칙 정의 (사용자 제공 파일명 기반)
        mapping = {
            "speed_plot": f"{prefix}_kinematics_speed.png",
            "trajectory_plot": f"{prefix}_kinematics_trajectory_speed.png", # 경로 예시 반영
            "trajectory_behavior": f"{prefix}_kinematics_trajectory_behavior.png",
            "turning_rate_plot_path": f"{prefix}_kinematics_turning_rate.png",
            "trajectory_turning_plot": f"{prefix}_kinematics_trajectory_turning.png",
            "nop_plot": f"{prefix}_kinematics_nop.png"
        }

        for key, filename in mapping.items():
            full_path = plots_dir / filename
            # NOP의 경우 하위 폴더에 있을 수 있으므로 체크
            if not full_path.exists():
                alt_path = base_dir / "nop" / filename
                if alt_path.exists(): full_path = alt_path

            # 파일이 실제로 존재하면 세션에 강제 업데이트
            if full_path.exists():
                st.session_state[key] = str(full_path)
            else:
                # 파일이 없으면 혹시 모르니 패턴 검색 시도
                found = _get_latest_file([plots_dir / f"{prefix}*{key.split('_')[0]}*.png"])
                if found: st.session_state[key] = found

init_session_state()

# --- [2] 사이드바: 프로젝트 관리 ---
# --- [2] Project Setup Page ---
def render_project_setup_page():
    st.title("Project & Config Setup")

    projects = list_projects(PROJECTS_ROOT)
    has_projects = bool(projects)

    with st.container(border=True):
        st.subheader("Select or Create Project")

        selected_proj = st.selectbox(
            "Existing Projects",
            options=projects if has_projects else ["(no projects yet)"],
            index=projects.index(st.session_state.active_project) if st.session_state.active_project in projects else 0,
            disabled=not has_projects,
        )

        new_name = st.text_input("New Project Name").strip()

        if st.button("Create Project", use_container_width=True) and new_name:
            create_project(PROJECTS_ROOT, new_name)
            st.session_state.active_project = new_name
            reset_analysis_state()
            st.rerun()

        if has_projects and selected_proj != st.session_state.active_project:
            st.session_state.active_project = selected_proj
            reset_analysis_state()

    if st.session_state.active_project and st.session_state.active_project != "(no projects yet)":
        with st.container(border=True):
            st.subheader("DLC Config")

            prof = load_profile(PROJECTS_ROOT, st.session_state.active_project)
            st.session_state.active_profile = prof

            cfg_up = st.file_uploader("Upload config.yaml", type=["yaml", "yml"])
            if st.button("Apply Uploaded Config", use_container_width=True) and cfg_up:
                import_dlc_config_file(PROJECTS_ROOT, st.session_state.active_project, cfg_up)
                st.rerun()

            cur_cfg = prof.get("dlc", {}).get("config_path", "")
            new_cfg_path = st.text_input("Direct Path", value=cur_cfg)
            if st.button("Update Path", use_container_width=True):
                set_dlc_config(PROJECTS_ROOT, st.session_state.active_project, new_cfg_path)
                st.rerun()

            st.caption(f"Status: {'Connected' if os.path.exists(cur_cfg) else 'Not Found'}")

        # Experiment type & camera settings
        with st.container(border=True):
            st.subheader("Experiment Type")
            st.session_state.experiment_type = st.selectbox(
                "Select Experiment Type",
                ["Open Field", "NOP", "VR", "Maze"],
                index=["Open Field", "NOP", "VR", "Maze"].index(st.session_state.experiment_type)
                if st.session_state.experiment_type in ["Open Field", "NOP", "VR", "Maze"] else 0,
            )

            st.subheader("Camera Setting")
            st.session_state.camera_view = st.selectbox(
                "Select Camera View",
                ["Top View", "Side", "Down"],
                index=["Top View", "Side", "Down"].index(st.session_state.camera_view)
                if st.session_state.camera_view in ["Top View", "Side", "Down"] else 0,
            )



        if st.button("Go to Main Page", type="primary", use_container_width=True):
            st.session_state.page = "main"
            st.rerun()

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
        
        cols = st.sidebar.columns(2)
        if cols[0].button("🎯 ROIs", use_container_width=True):
            st.session_state.page = "roi"; st.rerun()
        if cols[1].button("📝 Label", use_container_width=True):
            st.session_state.page = "annotate"; st.rerun()

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
        
        # ROI 분석 결과가 있다면 상단에 강조
        if st.session_state.roi_result_plot:
            with st.container(border=True):
                st.subheader("📍 ROI Performance Metrics")
                ri_c1, ri_c2 = st.columns([3, 2])
                ri_c1.image(st.session_state.roi_result_plot)
                ri_c2.dataframe(st.session_state.roi_result_df, use_container_width=True)

        # 탭을 이용한 플롯 분류
        tab1, tab2, tab3 = st.tabs(["📈 Basic Kinematics", "🔄 Turning Analysis", "🎭 Behavior & NOP"])
        
        with tab1:
            tc1, tc2 = st.columns(2)
            if st.session_state.speed_plot and os.path.exists(st.session_state.speed_plot): 
                tc1.image(st.session_state.speed_plot, caption="Speed Over Time")
            if st.session_state.trajectory_plot and os.path.exists(st.session_state.trajectory_plot):
                tc2.image(st.session_state.trajectory_plot, caption="Full Trajectory")
        
        with tab2:
            tc1, tc2 = st.columns(2)
            if st.session_state.turning_rate_plot_path:
                tc1.image(st.session_state.turning_rate_plot_path, caption="Turning Rate")
            if st.session_state.trajectory_turning_plot:
                tc2.image(st.session_state.trajectory_turning_plot, caption="Turning Trajectory")
        
        with tab3:
            tc1, tc2 = st.columns(2)
            if st.session_state.trajectory_behavior:
                tc1.image(st.session_state.trajectory_behavior, caption="Behavioral Map")
            if st.session_state.nop_plot:
                tc2.image(st.session_state.nop_plot, caption="NOP Analysis")

        with st.expander("📄 Detailed System Logs"):
            st.code("\n".join(st.session_state.logs))
else:
    st.info("Please upload a rodent video to begin.")
