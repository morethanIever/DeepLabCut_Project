# app.py
import os
import streamlit as st
from pipeline.run_pipeline import run_full_pipeline
st.set_page_config(page_title="Rodent Kinematics Analyzer", layout="wide")
st.title("Rodent Kinematics Analyzer 🐭")
st.write("Upload a video → Pose estimation → Kinematics → Behavior → Annotated video")

# ----------------------------
# Session state init
# ----------------------------
if "output_video" not in st.session_state:
    st.session_state.output_video = None

if "logs" not in st.session_state:
    st.session_state.logs = []
    
if "speed_plot" not in st.session_state:
    st.session_state.speed_plot = None

if "trajectory_plot" not in st.session_state:
    st.session_state.trajectory_plot = None

if "trajectory_behavior" not in st.session_state:
    st.session_state.trajectory_behavior = None

if "turning_rate_plot_path" not in st.session_state:
    st.session_state.turning_rate_plot_path = None
    
if "trajetory_turning_plot" not in st.session_state:
    st.session_state.trajectory_turning_plot = None
    
if "nop_plot" not in st.session_state:
    st.session_state.nop_plot = None

# ----------------------------
# File uploader
# ----------------------------
uploaded = st.file_uploader(
    "Upload a rodent video",
    type=["mp4", "avi", "mov", "mkv"]
)

# ----------------------------
# Cache options
# ----------------------------
with st.expander("Advanced options (cache control)", expanded=False):
    force_pose = st.checkbox("Force pose estimation again (slow)", value=False)
    force_analysis = st.checkbox("Recompute kinematics / behavior", value=False)

col1, col2 = st.columns(2)

# ----------------------------
# Input preview
# ----------------------------
if uploaded is not None:
    with col1:
        st.subheader("Input")
        st.video(uploaded)

# ----------------------------
# Analyze button
# ----------------------------
if uploaded is not None and st.button("Analyze Video", type="primary"):
    os.makedirs("temp", exist_ok=True)

    logs = []

    with st.spinner("Running the full pipeline..."):
        try:
            results = run_full_pipeline(
                uploaded,
                logs,
                force_pose=force_pose,
                force_analysis=force_analysis,
            )
            # 🔑 SAVE RESULTS IN SESSION STATE
            st.session_state.output_video = results["out_video"]
            st.session_state.logs = results["logs"]
            st.session_state.speed_plot = results["speed_plot"]
            st.session_state.trajectory_plot = results["trajectory_plot"]
            st.session_state.trajectory_behavior = results["trajectory_behavior"]
            st.session_state.turning_rate_plot_path = results["turning_rate_plot_path"]
            st.session_state.trajectory_turning_plot = results["trajectory_turning_plot"]
            st.session_state.nop_plot = results["nop_plot"]

        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

# ----------------------------
# Output section (persistent)
# ----------------------------
if st.session_state.output_video is not None:
    with col2:
        st.subheader("Output (Annotated)")
        st.video(st.session_state.output_video)

        with open(st.session_state.output_video, "rb") as f:
            st.download_button(
                label="Download annotated video",
                data=f,
                file_name=os.path.basename(st.session_state.output_video),
                mime="video/mp4",
            )

    with st.expander("Logs"):
        st.code("\n".join(st.session_state.logs))
        
   


    if st.session_state.speed_plot is not None:
        st.subheader("Speed over time")
        st.image(st.session_state.speed_plot, width=700)
    
    if st.session_state.trajectory_plot is not None:
        st.subheader("trajectory")
        st.image(st.session_state.trajectory_plot, width=700)
        
    if st.session_state.trajectory_behavior is not None:
        st.subheader("trajectory by behaviour")
        st.image(st.session_state.trajectory_behavior, width=700)
        
    if st.session_state.turning_rate_plot_path is not None:
        st.subheader("turning rate")
        st.image(st.session_state.turning_rate_plot_path, width=700)
    
    if st.session_state.trajectory_turning_plot is not None:
        st.subheader("trajectory + turning rate")
        st.image(st.session_state.trajectory_turning_plot, width=700)
        
    if st.session_state.nop_plot is not None:
        st.subheader("nop")
        st.image(st.session_state.nop_plot, width=700)