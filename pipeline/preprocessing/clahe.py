import cv2
import os
import subprocess
import time

def apply_clahe_to_video(input_path, output_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    temp_output = output_path.replace(".mp4", "_raw.mp4")
    
    # Use a function scope or explicit release
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l2 = clahe.apply(l)
            lab = cv2.merge((l2, a, b))
            enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            out.write(enhanced_frame)
    finally:
        # 1. Release everything explicitly
        cap.release()
        out.release()
        # 2. Force delete the objects to clear the Windows file handle
        del cap
        del out
        # 3. Give Windows a tiny moment to breathe
        time.sleep(1)

    # Now FFmpeg can safely access the file
    if os.path.exists(temp_output):
        subprocess.run([
            'ffmpeg', '-y', '-i', temp_output,
            '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
            '-crf', '18', output_path
        ], check=True)
        
        # Final cleanup
        try:
            os.remove(temp_output)
        except OSError:
            pass # If it still fails, it's just a temp file left behind
            
    return output_path