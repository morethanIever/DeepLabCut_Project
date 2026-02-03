# app.py
import os
import streamlit as st
import hashlib
import cv2
from PIL import Image
import traceback
from streamlit_drawable_canvas import st_canvas
from pipeline.run_pipeline import run_full_pipeline
from pipeline.preprocessing.crop import select_crop_roi, apply_crop
from pipeline.preprocessing.downsmaple import apply_downsample
from pipeline.ROI.ROI_anlaysis import run_multi_roi_analysis
from pipeline.ROI.canvasToROI import canvas_objects_to_rois
from pipeline.ROI.ROItoCanvas import rois_to_canvas_json
from pipeline.ROI.ROIEditor import render_roi_editor

st.set_page_config(page_title="Rodent Kinematics Analyzer", layout="wide")
st.title("Rodent Kinematics Analyzer 🐭")
st.write("Upload a video → Pose estimation → Kinematics → Behavior → Annotated video")
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# ----------------------------
# Session state init
# ----------------------------
if "input_video_path" not in st.session_state:
    st.session_state.input_video_path = None

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

if "crop_roi" not in st.session_state:
    st.session_state.crop_roi = None

if "resize_to" not in st.session_state:
    st.session_state.resize_to = None

if "kin_csv_path" not in st.session_state:
    st.session_state.kin_csv_path = None

if "roi_canvas_version" not in st.session_state:
    st.session_state.roi_canvas_version = 0

if "page" not in st.session_state:
    st.session_state.page = "main"

if "roi_list" not in st.session_state:
    st.session_state.roi_list = []
    
if "roi_radius" not in st.session_state:
    st.session_state.roi_radius = 80

if "roi_result_df" not in st.session_state:
    st.session_state.roi_result_df = None

if "roi_result_plot" not in st.session_state:
    st.session_state.roi_result_plot = None


# ----------------------------
# File uploader
# ----------------------------
uploaded = st.file_uploader(
    "Upload a rodent video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded is not None:
    # --- Detect new upload using content hash ---
    file_bytes = uploaded.getbuffer()
    file_id = hashlib.md5(file_bytes).hexdigest()

    if st.session_state.get("uploaded_file_id") != file_id:
        # New video → reset dependent state
        st.session_state.uploaded_file_id = file_id
        st.session_state.input_video_path = None
        st.session_state.output_video = None
        st.session_state.logs = []
        st.session_state.speed_plot = None
        st.session_state.trajectory_plot = None
        st.session_state.trajectory_behavior = None
        st.session_state.turning_rate_plot_path = None
        st.session_state.trajectory_turning_plot = None
        st.session_state.nop_plot = None
        st.session_state.crop_roi = None
        st.session_state.resize_to = None
        st.session_state.roi = None   # also reset temporary ROI selector


    # Save uploaded file (always when new)
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    initial_path = os.path.join(temp_dir, uploaded.name)

    if st.session_state.input_video_path is None:
        with open(initial_path, "wb") as f:
            f.write(file_bytes)
        st.session_state.input_video_path = initial_path

    # --- PREPROCESSING SUB-MENU ---
    st.markdown("### 🛠 Preprocessing")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        with st.expander("✂ Crop Video", expanded=False):
            st.write("Click the button")
            st.caption("Drag a rectangle, then press ENTER/SPACE. Press C to cancel.")

            if st.button("Open Crop GUI", key="open_crop_gui_btn"):
                roi = select_crop_roi(st.session_state.input_video_path)

                if roi is None:
                    st.warning("Crop cancelled or no ROI selected.")
                else:
                    st.session_state.crop_roi = roi
                    st.success(f"ROI selected: {roi} (x, y, w, h)")

            # ✅ Correct key: crop_roi
            if st.session_state.get("crop_roi") is not None:
                st.info(f"Selected ROI: {st.session_state.crop_roi}")

                if st.button("Apply Crop", key="apply_crop_btn"):
                    x, y, w, h = st.session_state.crop_roi

                    clean_name = "".join([c for c in uploaded.name if c.isalnum() or c in "._-"])
                    cropped_path = os.path.abspath(os.path.join("temp", f"cropped_{clean_name}"))

                    with st.spinner("Cropping with FFmpeg..."):
                        apply_crop(st.session_state.input_video_path, cropped_path, (x, y, w, h))

                    # ✅ update current video to cropped file
                    st.session_state.input_video_path = cropped_path

                    st.success("Cropped successfully!")

                    cap = cv2.VideoCapture(st.session_state.input_video_path)
                    st.write("W,H:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    cap.release()

                    st.rerun()





    with c2:
        with st.expander("⏱ Shorten"):
            t_start = st.text_input("Start time (seconds or HH:MM:SS)", value="0")
            t_end = st.text_input("End time (seconds or HH:MM:SS)", value="10")
            
            if st.button("Confirm Trim"):
                # Use a sanitized name
                clean_name = "".join([c for c in uploaded.name if c.isalnum() or c in "._-"])
                trimmed_path = os.path.abspath(os.path.join(temp_dir, f"trim_{clean_name}"))
                
                from pipeline.preprocessing.trim import apply_trim
                try:
                    with st.spinner("Trimming video..."):
                        apply_trim(st.session_state.input_video_path, trimmed_path, t_start, t_end)
                    
                    st.session_state.input_video_path = trimmed_path
                    st.success(f"Trimmed: {os.path.basename(trimmed_path)}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Trimming failed: {e}")

    with c3:
        with st.expander("📉 Downsample"):
            st.write("Reduce resolution to speed up DLC")
            # Common rodent video aspect ratios
            target_w = st.number_input("Target Width", value=1928, step=2)
            target_h = st.number_input("Target Height", value=1024, step=2)
            
            if st.button("Confirm Downsample"):
                
                # Sanitize filename
                clean_name = "".join([c for c in uploaded.name if c.isalnum() or c in "._-"])
                downsampled_path = os.path.abspath(os.path.join(temp_dir, f"down_{target_w}x{target_h}_{clean_name}"))
                
                try:
                    with st.spinner(f"Resizing to {target_w}x{target_h}..."):
                        apply_downsample(
                            st.session_state.input_video_path, 
                            downsampled_path, 
                            target_w, 
                            target_h
                        )
                    
                    # Update the state so the pipeline uses this new file
                    st.session_state.input_video_path = downsampled_path
                    st.session_state.resize_to = (target_w, target_h)
                    st.success("Video Resized!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Downsampling failed: {e}")

    with c4:
            with st.expander("✨ Enhance"):
                st.write("CLAHE Contrast Enhancement")
                st.caption("Best for Long Evans rats in black/white arenas.")
                c_limit = st.slider("Contrast Limit", 1.0, 5.0, 2.0)
                
                if st.button("Apply Enhancement"):
                    from pipeline.preprocessing.clahe import apply_clahe_to_video
                    
                    clean_name = "".join([c for c in uploaded.name if c.isalnum() or c in "._-"])
                    enhanced_path = os.path.abspath(os.path.join(temp_dir, f"enhanced_{clean_name}"))
                    
                    try:
                        with st.spinner("Enhancing contrast... (this may take a minute)"):
                            apply_clahe_to_video(
                                st.session_state.input_video_path, 
                                enhanced_path,
                                clip_limit=c_limit
                            )
                        
                        st.session_state.input_video_path = enhanced_path
                        st.success("Contrast Enhanced!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Enhancement failed: {e}")


    # Display the current version of the video being used
    st.info(f"Currently using: {os.path.basename(st.session_state.input_video_path)}")
    st.video(st.session_state.input_video_path)

# ----------------------------
# Cache options
# ----------------------------
with st.expander("Advanced options (cache control)", expanded=False):
    force_pose = st.checkbox("Force pose estimation again (slow)", value=False)
    force_analysis = st.checkbox("Recompute kinematics / behavior", value=False)

#Extract outliers
with st.expander("Pose QC (Quality Control)", expanded=False):
    st.warning("⚠ This will extract outlier frames for re-labeling (DLC)")
    qc_btn = st.button("🔧 Extract Pose Outliers")
    if qc_btn:
        with st.spinner("Extracting outlier frames using DeepLabCut..."):
            try:
                import deeplabcut as dlc

                dlc.extract_outlier_frames(
                    config=r"C:\Users\leelab\Desktop\TestBehaviour-Eunhye-2025-12-29\config.yaml",
                    videos=[st.session_state.input_video_path],
                    shuffle=1,
                    trainingsetindex=0,
                    #engine="pytorch",
                )

                st.success("✅ Outlier frames extracted successfully!")
                st.info(
                    "Next steps:\n"
                    "1) Open DLC GUI\n"
                    "2) Label extracted frames\n"
                    "3) Merge datasets & retrain model"
                )

            except Exception as e:
                st.error(f"Pose QC failed: {e}")

col1, col2 = st.columns(2)


# ----------------------------
# Input preview
# ----------------------------
if uploaded is not None:
    with col1:
        st.subheader("Input (Current)")
        # Show the processed path (cropped) instead of the raw upload
        st.video(st.session_state.input_video_path)

# ----------------------------
# Analyze button
# ----------------------------
if uploaded is not None and st.button("Analyze Video", type="primary"):
    # Always use the path that has been updated by preprocessing
    current_video = st.session_state.input_video_path
    roi = st.session_state.get("crop_roi", None)
    resize_to = st.session_state.get("resize_to", None)
    logs = []

    with st.spinner("Running the full pipeline..."):
        try:
            results = run_full_pipeline(
                current_video,
                logs,
                force_pose=force_pose,
                force_analysis=force_analysis,
                roi=roi,
                resize_to=resize_to
            )
            # 🔑 SAVE RESULTS IN SESSION STATE
            st.session_state.input_video_path = results["input_video"]
            st.session_state.kin_csv_path=results["kin_csv"]
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
            st.code(traceback.format_exc()) 
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
        

# ----------------------------
# Page routing (AFTER function definitions)
# ----------------------------
if st.session_state.page == "roi":
    render_roi_editor()
    st.stop()

st.sidebar.header("📍 ROI Analysis")

if st.sidebar.button("🎯 ROI Analyze (Draw ROIs)"):
    st.session_state.page = "roi"
    st.rerun()

st.sidebar.write(f"Saved ROIs: {len(st.session_state.roi_list)}")
st.sidebar.write(f"Radius: {st.session_state.roi_radius}px")

if st.sidebar.button("▶ Run ROI Analysis"):
    if st.session_state.kin_csv_path is None:
        st.sidebar.error("먼저 Analyze Video를 실행해서 kin_csv_path를 만들어야 해요.")
    elif not st.session_state.roi_list:
        st.sidebar.error("ROI가 없습니다. ROI Analyze에서 ROI를 그려주세요.")
    else:
        summary_df, plot_path = run_multi_roi_analysis(
            kin_csv=st.session_state.kin_csv_path,
            roi_list=st.session_state.roi_list,
            radius=st.session_state.roi_radius,
            fps=30,
            out_dir="outputs/roi",
        )
        st.session_state.roi_result_df = summary_df
        st.session_state.roi_result_plot = plot_path
        st.rerun()

if st.session_state.roi_result_plot is not None:
    st.subheader("📊 ROI Analysis Result")
    st.image(st.session_state.roi_result_plot)
    st.dataframe(st.session_state.roi_result_df)


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


