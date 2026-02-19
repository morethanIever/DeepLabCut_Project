import matplotlib.pyplot as plt
import pandas as pd
import os
    
def plot_nop(
    kin_csv: str,
    nop_summary_csv:str,
    object_left=(480, 130),
    object_right=(957, 130),
    arena_size=(1280, 720),
    out_path="outputs/plots/nop_validation.png"
):

    df = pd.read_csv(kin_csv)
    summary = pd.read_csv(nop_summary_csv)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(6, 4))

    # trajectory (skip if nose is missing)
    if "nose_x" in df.columns and "nose_y" in df.columns:
        plt.plot(df["nose_x"].values, df["nose_y"].values, alpha=0.3, linewidth=1)

    # objects
    plt.scatter(*object_left, s=2000, facecolors='none', edgecolors='r', label="Object A")
    plt.scatter(*object_right, s=2000, facecolors='none', edgecolors='b', label="Object B")

    plt.xlim(0, arena_size[0])
    plt.ylim(arena_size[1], 0)  # invert y
    
    nop_index = float(summary["NOP_index"].iloc[0])
    plt.title(f"NOP validation (NOP index = {nop_index:.2f})")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return out_path
