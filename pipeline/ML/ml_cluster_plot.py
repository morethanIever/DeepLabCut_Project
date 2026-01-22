import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_ml_clusters(
    kin_csv: str,
    cluster_csv: str,
    ml_features_csv: str,  
    out_dir: str = "outputs/plots/ml"
):
    os.makedirs(out_dir, exist_ok=True)

    kin = pd.read_csv(kin_csv)
    cl = pd.read_csv(cluster_csv)
    feats = pd.read_csv(ml_features_csv)

    if len(kin) != len(cl):
        raise RuntimeError(f"Frame mismatch: kin={len(kin)}, cluster={len(cl)}")

    # 데이터 병합 (분석용)
    merged = pd.concat([cl, feats], axis=1)

    # 1️⃣ PCA space plot (Behavior Clusters)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=cl, x="pc1", y="pc2", hue="cluster", 
        palette="tab10", s=10, alpha=0.6, edgecolor=None
    )
    plt.title("Rat Behavior Clusters in PCA Space")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "01_cluster_pca.png"), dpi=150)
    plt.close()

    # 2️⃣ Cluster Feature Profile (핵심: 행동 해석용 히트맵)
    # 클러스터별로 피처들의 평균을 내어 어떤 행동인지 보여줍니다.
    plt.figure(figsize=(10, 6))
    summary_norm = merged.groupby("cluster").mean().drop(columns=["pc1", "pc2"])
    # 가독성을 위해 피처별로 0~1 사이로 정규화하여 히트맵 생성
    summary_scaled = (summary_norm - summary_norm.min()) / (summary_norm.max() - summary_norm.min())
    sns.heatmap(summary_scaled, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Feature Intensity by Cluster (Normalized 0-1)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "02_feature_heatmap.png"), dpi=150)
    plt.close()

    # 3️⃣ Distribution of Key Features (Boxplots)
    # 주요 행동 지표들의 분포 확인 (speed, body_stretch 등)
    key_cols = ["speed", "body_stretch", "abs_angular_velocity", "head_to_body_dist"]
    # 데이터에 존재하는 컬럼만 선택
    available_cols = [c for c in key_cols if c in merged.columns]
    
    if available_cols:
        fig, axes = plt.subplots(1, len(available_cols), figsize=(15, 4))
        for i, col in enumerate(available_cols):
            sns.boxplot(data=merged, x="cluster", y=col, ax=axes[i], palette="tab10")
            axes[i].set_title(f"Distribution: {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "03_feature_distribution.png"), dpi=150)
        plt.close()

    # 4️⃣ Trajectory colored by cluster (Spatial view)
    plt.figure(figsize=(6, 6))
    plt.scatter(
        kin["spine_x"], kin["spine_y"], 
        c=cl["cluster"], s=2, cmap="tab10", alpha=0.5
    )
    plt.gca().invert_yaxis() # Top-view 카메라 좌표계 보정
    plt.xlabel("X (px)")
    plt.ylabel("Y (px)")
    plt.title("Rat Trajectory by Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "04_cluster_trajectory.png"), dpi=150)
    plt.close()

    # 5️⃣ Cluster Ethogram (Time-series)
    plt.figure(figsize=(15, 3))
    plt.scatter(range(len(cl)), cl["cluster"], c=cl["cluster"], cmap="tab10", s=1, alpha=1)
    plt.xlabel("Frame")
    plt.ylabel("Cluster ID")
    plt.title("Ethogram (Behavior Sequence)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "05_cluster_ethogram.png"), dpi=150)
    plt.close()

    # 6️⃣ Summary CSV Save
    summary_norm.to_csv(os.path.join(out_dir, "cluster_feature_mean_summary.csv"))

    return f"Plots saved to {out_dir}"

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python -m pipeline.ML.ml_cluster_plot <kin.csv> <clusters.csv> <features.csv>")
        sys.exit(1)
    
    plot_ml_clusters(sys.argv[1], sys.argv[2], sys.argv[3])