import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_speed(kin_csv: str) -> str:
    df = pd.read_csv(kin_csv)

    frames = df["frame"]
    speed = df["speed_px_s"]

    plt.figure(figsize=(6, 3))
    plt.plot(frames, speed, color="blue", linewidth=1)
    plt.xlabel("Frame")
    plt.ylabel("Speed (px/s)")
    plt.title("Rodent Speed Over Time")
    plt.grid(alpha=0.3)

    os.makedirs("outputs/plots", exist_ok=True)
    out_path = "outputs/plots/speed.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return out_path
