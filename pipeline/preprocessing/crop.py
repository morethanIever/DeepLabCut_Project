import cv2
import subprocess
import os

def select_crop_roi(video_path):
    """Opens a local window to select the crop area."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None

    # Open OpenCV ROI Selector
    # Select area and press ENTER or SPACE. Press C to cancel.
    roi = cv2.selectROI("Select Crop Area & Press ENTER", frame, showCrosshair=True)
    cv2.destroyAllWindows()
    
    # roi is (x, y, w, h). If user cancels, it returns (0,0,0,0)
    return roi if roi != (0, 0, 0, 0) else None

def apply_crop(input_path, output_path, roi):
    """Applies the crop using FFmpeg (SimBA style)."""
    x, y, w, h = roi
    # Command: ffmpeg -i input -vf crop=w:h:x:y -c:v libx264 -crf 18 output
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-vf', f'crop={w}:{h}:{x}:{y}',
        '-c:v', 'libx264', '-crf', '18',
        output_path
    ]
    subprocess.run(cmd, check=True)
    return output_path