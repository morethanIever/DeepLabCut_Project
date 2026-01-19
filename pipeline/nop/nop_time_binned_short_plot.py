import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_nop_time_binned_short(
    csv_path: str,
    out_path="outputs/plots/nop_time_binned_short.png",
):
    df = pd.read_csv(csv_path)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x = range(len(df))

    plt.figure(figsize=(6, 4))

    plt.bar(x, df["left_exploration_s"], width=0.35, label="Left", color="#4C72B0")
    plt.bar(
        x,
        df["right_exploration_s"],
        width=0.35,
        bottom=df["left_exploration_s"],
        label="Right",
        color="#DD8452",
    )

    plt.plot(x, df["NOP_index"], "-o", color="black", linewidth=2)

    plt.xticks(x, df["bin"])
    plt.axhline(0, linestyle="--", color="gray")
    plt.ylabel("Exploration time (s) / NOP index")
    plt.title("NOP (short video, relative bins)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path
