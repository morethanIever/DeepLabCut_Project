def rois_to_canvas_json(roi_list):
    objects = []
    for r in roi_list:
        if r["type"] == "circle":
            rad = float(r["radius"])
            cx = float(r["cx"]); cy = float(r["cy"])
            objects.append({
                "id": r["id"],
                "type": "circle",
                "left": cx - rad,
                "top":  cy - rad,
                "radius": rad,
                "scaleX": 1.0,
                "scaleY": 1.0,
                "fill": "rgba(0,255,0,0.15)",
                "stroke": "rgba(0,255,0,0.9)",
                "strokeWidth": 2,
            })

        elif r["type"] == "rect":
            objects.append({
                "id": r["id"],
                "type": "rect",
                "left": float(r["left"]),
                "top": float(r["top"]),
                "width": float(r["w"]),
                "height": float(r["h"]),
                "scaleX": 1.0,
                "scaleY": 1.0,
                "fill": "rgba(255,0,0,0.10)",
                "stroke": "rgba(255,0,0,0.9)",
                "strokeWidth": 2,
            })

    return {"version": "4.4.0", "objects": objects}
