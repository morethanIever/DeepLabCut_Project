from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from PIL import Image

def canvas_objects_to_rois(objs):
    """
    streamlit-drawable-canvas object list -> roi_list
    roi format:
      {
        "id": "roi_1",
        "name": "ROI_1",
        "type": "circle" | "rect",
        "cx": float,
        "cy": float,
        "radius": float | None,
        "w": float | None,
        "h": float | None,
        "left": float,
        "top": float
      }
    """
    rois = []
    idx = 1

    for o in objs or []:
        t = o.get("type")

        if t == "circle":
            r = float(o.get("radius", 0))
            left = float(o.get("left", 0))
            top = float(o.get("top", 0))
            cx = left + r
            cy = top + r

            rois.append({
                "id": f"roi_{idx}",
                "name": f"ROI_{idx}",
                "type": "circle",
                "cx": cx, "cy": cy,
                "radius": r,
                "w": None, "h": None,
                "left": left, "top": top,
            })
            idx += 1

        elif t == "rect":
            left = float(o.get("left", 0))
            top = float(o.get("top", 0))
            w = float(o.get("width", 0))
            h = float(o.get("height", 0))
            cx = left + w / 2.0
            cy = top + h / 2.0

            rois.append({
                "id": f"roi_{idx}",
                "name": f"ROI_{idx}",
                "type": "rect",
                "cx": cx, "cy": cy,
                "radius": None,
                "w": w, "h": h,
                "left": left, "top": top,
            })
            idx += 1

    return rois