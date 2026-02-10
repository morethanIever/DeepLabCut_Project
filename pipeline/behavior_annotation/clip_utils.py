import os
import subprocess

FFMPEG = os.environ.get(
    "FFMPEG_PATH",
    r"C:\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
)

def save_video_clip_ffmpeg(video_path, start_frame, end_frame, out_path, fps=30.0):
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    if end_frame <= start_frame:
        end_frame = start_frame + 1

    fps = float(fps) if fps else 30.0
    start_sec = start_frame / fps
    duration_sec = (end_frame - start_frame + 1) / fps

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cmd = [
        FFMPEG, "-y",
        "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_sec:.6f}",
        "-i", video_path,
        "-t", f"{duration_sec:.6f}",
        "-an",

        # ✅ 브라우저/Streamlit 호환 최우선 세팅
        "-vf", f"fps={fps},format=yuv420p",
        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-level", "3.0",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "23",
        "-movflags", "+faststart",

        out_path
    ]

    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{p.stderr}\nCMD: {' '.join(cmd)}")

    if (not os.path.exists(out_path)) or os.path.getsize(out_path) < 1024:
        raise RuntimeError("Clip file not created or too small.")

    return out_path
