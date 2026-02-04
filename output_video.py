import os
import subprocess

FFMPEG = os.environ.get(
    "FFMPEG_PATH",
    r"C:\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
)

def make_streamlit_playable_mp4(in_path: str, out_path: str, fps: float | None = None):
    in_path = os.path.abspath(in_path)
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ✅ out_path가 in_path와 같으면 tmp로 바꿔서 생성 후 교체
    same_path = os.path.normcase(in_path) == os.path.normcase(out_path)
    tmp_out = out_path + ".tmp.mp4" if same_path else out_path

    vf = "format=yuv420p"
    if fps:
        vf = f"fps={float(fps)},format=yuv420p"

    cmd = [
        FFMPEG, "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", in_path,
        "-an",
        "-vf", vf,
        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-level", "3.0",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "23",
        "-movflags", "+faststart",
        tmp_out
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{p.stderr}\nCMD: {' '.join(cmd)}")

    if (not os.path.exists(tmp_out)) or os.path.getsize(tmp_out) < 1024:
        raise RuntimeError("Playable mp4 not created or too small.")

    # ✅ 입력=출력인 경우: tmp를 최종 out_path로 교체
    if same_path:
        os.replace(tmp_out, out_path)

    return out_path
