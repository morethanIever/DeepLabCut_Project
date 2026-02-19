import subprocess
import os

def apply_downsample(input_path, output_path, width, height):
    """
    Downsamples video resolution using FFmpeg.
    """
    # -vf scale: Changes resolution. 
    # -crf 18: High quality (SimBA default is usually around 23)
    command = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-vf', f'scale={width}:{height}',
        '-c:v', 'libx264',
        '-crf', '18',
        '-preset', 'veryfast',
        output_path
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Downsampling failed. FFmpeg error: {result.stderr}")
        
    return output_path