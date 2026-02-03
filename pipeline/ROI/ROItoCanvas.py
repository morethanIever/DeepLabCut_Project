import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas

def rois_to_canvas_json(roi_list, default_radius=80):
    """
    saved roi_list -> initial_drawing json (canvas에 기존 ROI를 다시 그려서 보여줌)
    """
    objects = []
    for r in roi_list:
        if r["type"] == "circle":
            rad = float(r.get("radius") or default_radius)
            objects.append({
                "type": "circle",
                "left": float(r["cx"]) - rad,
                "top": float(r["cy"]) - rad,
                "radius": rad,
                "fill": "rgba(0,255,0,0.15)",
                "stroke": "rgba(0,255,0,0.9)",
                "strokeWidth": 2,
            })
        elif r["type"] == "rect":
            objects.append({
                "type": "rect",
                "left": float(r["left"]),
                "top": float(r["top"]),
                "width": float(r["w"]),
                "height": float(r["h"]),
                "fill": "rgba(255,0,0,0.10)",
                "stroke": "rgba(255,0,0,0.9)",
                "strokeWidth": 2,
            })
    return {"version": "4.4.0", "objects": objects}
