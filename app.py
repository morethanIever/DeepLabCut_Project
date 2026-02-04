import os
import hashlib
import traceback
import streamlit as st

from pipeline.run_pipeline import run_full_pipeline
from pipeline.preprocessing.crop import select_crop_roi, apply_crop
from pipeline.preprocessing.downsmaple import apply_downsample
from pipeline.preprocessing.trim import apply_trim
from pipeline.preprocessing.clahe import apply_clahe_to_video
from pipeline.ROI.ROI_anlaysis import run_multi_roi_analysis
from pipeline.ROI.ROIEditor import render_roi_editor
from pipeline.behavior_annotation.annotator_ui import render_behavior_annotator_page
from output_video import make_streamlit_playable_mp4
# --- 기본 설정 ---
st.set_page_config(page_title="Rodent Kinematics Analyzer", layout="wide")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.makedirs("temp", exist_ok=True)


# --- 1) Session State 초기화 ---
def init_session_state():
    defaults = {
        "uploaded_file_id": None,
        "input_video_path": None,
        "output_video": None,
        "logs": [],
        "page": "main",
        "speed_plot": None,
        "trajectory_plot": None,
        "trajectory_behavior": None,
        "turning_rate_plot_path": None,
        "trajectory_turning_plot": None,
        "nop_plot": None,
        "crop_roi": None,
        "resize_to": None,
        "kin_csv_path": None,
        "roi_list": [],
        "roi_radius": 80,
        "roi_result_df": None,
        "roi_result_plot": None,
        # annotate 관련(있으면 유지)
        "ann_export_dir": "outputs/annotations",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_analysis_state_for_new_video():
    """
    새 비디오 업로드 시, 분석/ROI/플롯 등만 리셋.
    uploaded_file_id / input_video_path 같은 비디오 핵심키는 여기서 건드리지 않음.
    """
    keys_to_reset = [
        "output_video",
        "logs",
        "speed_plot",
        "trajectory_plot",
        "trajectory_behavior",
        "turning_rate_plot_path",
        "trajectory_turning_plot",
        "nop_plot",
        "crop_roi",
        "resize_to",
        "kin_csv_path",
        "roi_list",
        "roi_result_df",
        "roi_result_plot",
    ]
    for k in keys_to_reset:
        if k in st.session_state:
            if k == "logs":
                st.session_state[k] = []
            elif k == "roi_list":
                st.session_state[k] = []
            else:
                st.session_state[k] = None

    st.session_state.page = "main"


init_session_state()


# --- 2) 사이드바 네비게이션 ---
st.sidebar.title("🛠️ Tools & Navigation")

if st.sidebar.button("🏠 Home / Main Analysis"):
    st.session_state.page = "main"
    st.rerun()

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


# --- 3) 페이지 라우팅 ---
if st.session_state.page == "roi":
    render_roi_editor()
    st.stop()

elif st.session_state.page == "annotate":
    if not st.session_state.input_video_path or not os.path.exists(st.session_state.input_video_path):
        st.error("No video loaded. Go to Home and upload a video first.")
        st.stop()

    render_behavior_annotator_page(st.session_state.input_video_path)
    st.stop()


# --- 4) 메인 화면 ---
st.title("Rodent Kinematics Analyzer 🐭")

uploaded = st.file_uploader(
    "Upload a rodent video",
    type=["mp4", "avi", "mov", "mkv"],
    key="video_uploader"   # ✅ 업로더 안정화
)

# (A) 업로드 처리: uploaded가 있을 때만 저장/갱신
if uploaded is not None:
    file_bytes = uploaded.getbuffer()
    file_id = hashlib.md5(file_bytes).hexdigest()

    if st.session_state.get("uploaded_file_id") != file_id:
        # 새 비디오일 때만 분석 결과 리셋
        reset_analysis_state_for_new_video()

        st.session_state.uploaded_file_id = file_id

        temp_path = os.path.join("temp", uploaded.name)
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        st.session_state.input_video_path = temp_path


# (B) 메인 UI 렌더링: uploaded가 아니라 input_video_path 기준!
video_path = st.session_state.input_video_path

if video_path and os.path.exists(video_path):
    st.info(f"📁 Current Video: **{os.path.basename(video_path)}**")

    # --- 전처리 섹션 ---
    with st.expander("🛠 Preprocessing Tools", expanded=True):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            if st.button("✂ Open Crop GUI"):
                roi = select_crop_roi(video_path)
                if roi:
                    st.session_state.crop_roi = roi

            if st.session_state.crop_roi:
                st.caption(f"Selected: {st.session_state.crop_roi}")
                if st.button("Apply Crop"):
                    out_p = os.path.abspath(f"temp/cropped_{os.path.basename(video_path)}")
                    apply_crop(video_path, out_p, st.session_state.crop_roi)
                    st.session_state.input_video_path = out_p
                    st.rerun()

        with c2:
            t_start = st.text_input("Start (s)", "0")
            t_end = st.text_input("End (s)", "10")
            if st.button("Confirm Trim"):
                out_p = os.path.abspath(f"temp/trim_{os.path.basename(video_path)}")
                apply_trim(video_path, out_p, t_start, t_end)
                st.session_state.input_video_path = out_p
                st.rerun()

        with c3:
            tw = st.number_input("Width", 1928, step=2)
            th = st.number_input("Height", 1024, step=2)
            if st.button("Confirm Resize"):
                out_p = os.path.abspath(f"temp/down_{tw}x{th}_{os.path.basename(video_path)}")
                apply_downsample(video_path, out_p, tw, th)
                st.session_state.input_video_path = out_p
                st.session_state.resize_to = (tw, th)
                st.rerun()

        with c4:
            c_limit = st.slider("Contrast", 1.0, 5.0, 2.0)
            if st.button("Apply CLAHE"):
                out_p = os.path.abspath(f"temp/enhanced_{os.path.basename(video_path)}")
                apply_clahe_to_video(video_path, out_p, c_limit)
                st.session_state.input_video_path = out_p
                st.rerun()

    # --- 미리보기 ---
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.subheader("Input Preview")
        st.video(st.session_state.input_video_path)

    # --- 파이프라인 옵션 ---
    with st.expander("⚙️ Pipeline Options"):
        f_pose = st.checkbox("Force Pose Estimation")
        f_anal = st.checkbox("Force Kinematics Analysis")

        if st.button("🔧 Extract DLC Outliers"):
            import deeplabcut as dlc
            dlc.extract_outlier_frames(config="config.yaml", videos=[st.session_state.input_video_path])

    if st.button("🚀 Analyze Video", type="primary", use_container_width=True):
        with st.spinner("Processing full pipeline..."):
            try:
                res = run_full_pipeline(
                    st.session_state.input_video_path, [],
                    force_pose=f_pose,
                    force_analysis=f_anal,
                    roi=st.session_state.crop_roi,
                    resize_to=st.session_state.resize_to,
                )

                # 1) output video를 Streamlit 호환 mp4로 변환
                out_vid = res.get("out_video") or res.get("output_video")
                if out_vid and os.path.exists(out_vid):
                    playable_path = os.path.abspath(os.path.join("temp", "playable_output.mp4"))
                    playable_path = make_streamlit_playable_mp4(out_vid, playable_path, fps=None)
                    res["out_video"] = playable_path

                # 2) ✅ res 결과를 session_state에 반영 (이게 없어서 결과가 안 나왔던 것)
                key_map = {
                    "input_video": "input_video_path",
                    "kin_csv": "kin_csv_path",
                    "out_video": "output_video",
                    "output_video": "output_video",
                }
                for k, v in res.items():
                    st.session_state[key_map.get(k, k)] = v

                # 디버그: res에 뭐가 들어왔는지 확인용(원하면 나중에 제거)
                st.session_state["last_pipeline_keys"] = list(res.keys())

                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")
                st.code(traceback.format_exc())


    # --- 결과 전시 ---
    if st.session_state.output_video and os.path.exists(st.session_state.output_video):
        with col_v2:
            st.subheader("Analysis Result")
            st.video(st.session_state.output_video)
            with open(st.session_state.output_video, "rb") as f:
                st.download_button("📥 Download Result", f, file_name="annotated.mp4")

        st.markdown("---")
        st.header("📊 Kinematics Data Visualization")

        if st.session_state.roi_result_plot:
            st.subheader("ROI Analysis")
            st.image(st.session_state.roi_result_plot)
            st.dataframe(st.session_state.roi_result_df, use_container_width=True)

        plots = {
            "Speed": st.session_state.speed_plot,
            "Trajectory": st.session_state.trajectory_plot,
            "Behavioral Trajectory": st.session_state.trajectory_behavior, #지우기
            "Turning Rate": st.session_state.turning_rate_plot_path,
            "Turning Trajectory": st.session_state.trajectory_turning_plot,
            "NOP": st.session_state.nop_plot, #지우기
        }

        plot_cols = st.columns(2)
        for i, (title, path) in enumerate(plots.items()):
            if path:
                with plot_cols[i % 2]:
                    st.subheader(title)
                    st.image(path, use_column_width=True)

        with st.expander("📄 Process Logs"):
            st.code("\n".join(st.session_state.logs))

else:
    st.warning("No video loaded yet. Upload a video first.")
