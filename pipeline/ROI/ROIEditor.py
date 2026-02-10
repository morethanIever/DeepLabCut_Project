import cv2
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas



from pipeline.ROI.ROI_anlaysis import run_multi_roi_analysis
from pipeline.ROI.canvasToROI import canvas_objects_to_rois
from pipeline.ROI.ROItoCanvas import rois_to_canvas_json

def render_roi_editor():
    def _rerun():
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()
    st.title("🎯 ROI Editor")
    st.write("ROI를 그린 뒤 **Save & Back**을 누르면 메인 화면에서 분석할 수 있습니다.")

    if st.session_state.input_video_path is None:
        st.error("먼저 비디오를 업로드하세요.")
        if st.button("⬅ Back"):
            st.session_state.page = "main"
            _rerun()
        return

    # canvas reset key
    if "roi_canvas_rev" not in st.session_state:
        st.session_state.roi_canvas_rev = 0

    # selection state
    if "roi_delete_ids" not in st.session_state:
        st.session_state.roi_delete_ids = set()

    # background frame
    cap = cv2.VideoCapture(st.session_state.input_video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        st.error("비디오 첫 프레임을 읽지 못했습니다.")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bg = Image.fromarray(frame_rgb)

    # draw mode
    draw_mode = st.radio("Drawing mode", ["circle", "rect"], horizontal=True, key="roi_draw_mode")

    # canvas
    canvas_key = f"roi_canvas_{st.session_state.roi_canvas_rev}"
    canvas_result = st_canvas(
        background_image=bg,
        drawing_mode=draw_mode,
        update_streamlit=True,
        stroke_width=2,
        stroke_color="rgba(0,255,0,1.0)" if draw_mode == "circle" else "rgba(255,0,0,1.0)",
        fill_color="rgba(0,255,0,0.15)" if draw_mode == "circle" else "rgba(255,0,0,0.10)",
        height=bg.height,
        width=bg.width,
        key=canvas_key,
        initial_drawing=rois_to_canvas_json(st.session_state.roi_list),
    )

    # current ROIs computed from canvas
    objs = canvas_result.json_data.get("objects", []) if canvas_result.json_data else []
    current_rois = canvas_objects_to_rois(objs)

    st.write("---")
    st.subheader("🧾 ROI List (select to delete)")
    if not current_rois:
        st.info("아직 ROI가 없습니다. 캔버스에서 ROI를 그려주세요.")
    else:
        # show selectable list
        delete_ids = set(st.session_state.roi_delete_ids)

        for r in current_rois:
            key = f"chk_{r['id']}"
            checked = st.checkbox(
                f"{r['id']} | {r['type']} | (cx={r['cx']:.1f}, cy={r['cy']:.1f})",
                key=key,
                value=(r["id"] in delete_ids)
            )
            if checked:
                delete_ids.add(r["id"])
            else:
                delete_ids.discard(r["id"])

        st.session_state.roi_delete_ids = delete_ids

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            if st.button("🗑 Delete Selected"):
                kept = [r for r in current_rois if r["id"] not in st.session_state.roi_delete_ids]
                st.session_state.roi_list = kept
                st.session_state.roi_delete_ids = set()   # ✅ 삭제 후 선택 초기화
                st.session_state.roi_canvas_rev += 1
                _rerun()


        with c2:
            if st.button("🧹 Clear All ROIs"):
                st.session_state.roi_list = []
                st.session_state.roi_canvas_rev += 1
                _rerun()

        with c3:
            if st.button("💾 Save & Back"):
                # Save exactly what is currently on canvas (after deletes)
                st.session_state.roi_list = current_rois
                st.session_state.page = "main"
                _rerun()

        with c4:
            if st.button("⬅ Cancel"):
                st.session_state.page = "main"
                _rerun()
