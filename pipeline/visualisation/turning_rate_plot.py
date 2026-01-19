import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_turning_rate(turn_csv: str) -> str:
    df = pd.read_csv(turn_csv)

    plt.figure(figsize=(7, 3))
    plt.plot(df["frame"], df["turning_rate_deg_s"], color="purple", lw=1)
    plt.axhline(0, color="black", lw=0.5)

    plt.title("Turning Rate (deg/s)")
    plt.xlabel("Frame")
    plt.ylabel("deg/s")
    plt.grid(alpha=0.3)

    os.makedirs("outputs/plots", exist_ok=True)
    out_path = "outputs/plots/turning_rate.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return out_path
