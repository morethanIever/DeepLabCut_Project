import hashlib

def _stable_id(o: dict) -> str:
    t = (o.get("type") or "").lower()
    left = float(o.get("left", 0))
    top = float(o.get("top", 0))
    w = float(o.get("width", 0))
    h = float(o.get("height", 0))
    r = float(o.get("radius", 0))

    # scale까지 포함해야 더 안정적
    sx = float(o.get("scaleX", 1.0))
    sy = float(o.get("scaleY", 1.0))

    payload = f"{t}|{round(left,2)}|{round(top,2)}|{round(w,2)}|{round(h,2)}|{round(r,2)}|{round(sx,2)}|{round(sy,2)}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]


def canvas_objects_to_rois(objs):
    rois = []
    for o in objs:
        t = (o.get("type") or "").lower()
        rid = o.get("id") or _stable_id(o)

        if t == "circle":
            left = float(o.get("left", 0))
            top = float(o.get("top", 0))

            r0 = float(o.get("radius", 0))
            sx = float(o.get("scaleX", 1.0))
            sy = float(o.get("scaleY", 1.0))

            # ✅ 실제 픽셀 반지름으로 환산 (보통 sx==sy)
            radius = r0 * (sx + sy) / 2.0

            cx = left + radius
            cy = top + radius

            rois.append({
                "id": rid,
                "type": "circle",
                "cx": cx,
                "cy": cy,
                "radius": radius,  # ✅ 실제 반지름 저장
                "name": o.get("roi_name", f"ROI_{rid}")
            })

        elif t == "rect":
            left = float(o.get("left", 0))
            top = float(o.get("top", 0))

            w0 = float(o.get("width", 0))
            h0 = float(o.get("height", 0))
            sx = float(o.get("scaleX", 1.0))
            sy = float(o.get("scaleY", 1.0))

            # ✅ 실제 픽셀 너비/높이로 환산
            w = w0 * sx
            h = h0 * sy

            cx = left + w / 2.0
            cy = top + h / 2.0

            rois.append({
                "id": rid,
                "type": "rect",
                "left": left,
                "top": top,
                "w": w,
                "h": h,
                "cx": cx,
                "cy": cy,
                "name": o.get("roi_name", f"ROI_{rid}")
            })

    return rois
