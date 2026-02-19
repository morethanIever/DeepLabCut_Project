import subprocess
import os

def apply_trim(input_path, output_path, start_time, end_time):
    # -y: overwrite output
    # -ss: start time
    # -to: end time
    # -c:v libx264: re-encode to ensure the cut is precise
    command = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-ss', str(start_time),
        '-to', str(end_time),
        '-c:v', 'libx264', 
        '-crf', '18', 
        '-preset', 'veryfast',
        output_path
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"FFmpeg failed to create the file. Error: {result.stderr}")
        
    return output_path