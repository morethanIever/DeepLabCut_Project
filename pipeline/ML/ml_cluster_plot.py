import os
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# Main plotting function
# -------------------------------------------------
def plot_ml_clusters(
    kin_csv: str,
    cluster_csv: str,
    out_dir: str = "outputs/plots/ml"
):
    os.makedirs(out_dir, exist_ok=True)

    kin = pd.read_csv(kin_csv)
    cl  = pd.read_csv(cluster_csv)

    if len(kin) != len(cl):
        raise RuntimeError(
            f"Frame mismatch: kin={len(kin)}, cluster={len(cl)}"
        )

    # -------------------------------------------------
    # 1️⃣ PCA space plot
    # -------------------------------------------------
    plt.figure(figsize=(5, 5))
    sc = plt.scatter(
        cl["pc1"],
        cl["pc2"],
        c=cl["cluster"],
        s=6,
        cmap="tab10",
        alpha=0.8
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("ML behavior clusters (PCA space)")
    plt.colorbar(sc, label="Cluster ID")
    plt.tight_layout()

    pca_path = os.path.join(out_dir, "cluster_pca.png")
    plt.savefig(pca_path, dpi=150)
    plt.close()

    # -------------------------------------------------
    # 2️⃣ Cluster over time
    # -------------------------------------------------
    plt.figure(figsize=(12, 3))
    plt.plot(cl["cluster"], lw=0.7)
    plt.xlabel("Frame")
    plt.ylabel("Cluster ID")
    plt.title("Cluster assignment over time")
    plt.tight_layout()

    time_path = os.path.join(out_dir, "cluster_time.png")
    plt.savefig(time_path, dpi=150)
    plt.close()

    # -------------------------------------------------
    # 3️⃣ Trajectory colored by cluster
    # -------------------------------------------------
    plt.figure(figsize=(5, 5))
    sc = plt.scatter(
        kin["spine_x"],
        kin["spine_y"],
        c=cl["cluster"],
        s=5,
        cmap="tab10",
        alpha=0.8
    )
    plt.gca().invert_yaxis()
    plt.xlabel("X (px)")
    plt.ylabel("Y (px)")
    plt.title("Trajectory colored by ML cluster")
    plt.colorbar(sc, label="Cluster ID")
    plt.tight_layout()

    traj_path = os.path.join(out_dir, "cluster_trajectory.png")
    plt.savefig(traj_path, dpi=150)
    plt.close()

    # -------------------------------------------------
    # 4️⃣ Cluster-wise feature summary
    # -------------------------------------------------
    merged = pd.concat([cl, kin], axis=1)

    summary = (
        merged
        .groupby("cluster")[[
            "speed_px_s",
            "turning_rate_deg",
            "move_turn_angle_deg"
        ]]
        .mean()
        .round(2)
    )

    summary_path = os.path.join(out_dir, "cluster_feature_summary.csv")
    summary.to_csv(summary_path)

    # -------------------------------------------------
    # 5️⃣ Cluster distribution
    # -------------------------------------------------
    dist = cl["cluster"].value_counts(normalize=True).sort_index()

    plt.figure(figsize=(5, 3))
    dist.plot(kind="bar")
    plt.ylabel("Fraction")
    plt.xlabel("Cluster ID")
    plt.title("Cluster distribution")
    plt.tight_layout()

    dist_path = os.path.join(out_dir, "cluster_distribution.png")
    plt.savefig(dist_path, dpi=150)
    plt.close()

    return {
        "pca": pca_path,
        "time": time_path,
        "trajectory": traj_path,
        "summary_csv": summary_path,
        "distribution": dist_path,
    }


# -------------------------------------------------
# CLI entry
# -------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage:\n"
            "python -m pipeline.ML.ml_cluster_plot "
            "<kinematics.csv> <ml_clusters.csv>"
        )
        sys.exit(1)

    kin_csv = sys.argv[1]
    cluster_csv = sys.argv[2]

    paths = plot_ml_clusters(kin_csv, cluster_csv)

    print("[ML PLOT] Saved:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
