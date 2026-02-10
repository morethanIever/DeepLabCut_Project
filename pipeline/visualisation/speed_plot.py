import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_speed(kin_csv: str, out_dir: str) -> str:
    df = pd.read_csv(kin_csv)

    frames = df["frame"]
    speed = df["speed_px_s"]

    plt.figure(figsize=(6, 3))
    plt.plot(frames.to_numpy(), speed.to_numpy(), color="blue", linewidth=1)
    plt.xlabel("Frame")
    plt.ylabel("Speed (px/s)")
    plt.title("Rodent Speed Over Time")
    plt.grid(alpha=0.3)

    os.makedirs(out_dir, exist_ok=True)
    video_stem = os.path.splitext(os.path.basename(kin_csv))[0]
    out_path = os.path.join(out_dir, f"{video_stem}_speed.png")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return out_path
