import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def run_clustering(ml_csv: str, out_dir: str, n_clusters=7):
    df = pd.read_csv(ml_csv)

    # 0값이 너무 많은 경우를 대비해 무한대 값 처리 및 결측치 제거
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = df.values

    # 1) normalize
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)

    # 2) PCA (시각화용)
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(Xs)

    # 3) clustering
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = km.fit_predict(Xs)

    out = pd.DataFrame({
        "pc1": Xp[:, 0],
        "pc2": Xp[:, 1],
        "cluster": labels,
    })

    os.makedirs(out_dir, exist_ok=True)

    base = os.path.basename(ml_csv).replace("_features.csv", "")
    out_path = os.path.join(out_dir, f"{base}_clusters.csv")
    out.to_csv(out_path, index=False)

    print(f"[ML] Clusters saved to {out_path}")

    cluster_profile = df.groupby(labels).mean()
    print("\n[ML] Cluster Profiles (Mean values):")
    print(cluster_profile[['speed', 'body_stretch', 'turn_rate']]) # 주요 지표 위주 출력

    return out_path

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m pipeline.ML.ml_cluster <ml_features.csv> <out_dir>")
        sys.exit(1)

    ml_csv = sys.argv[1]
    out_dir = sys.argv[2]

    run_clustering(
        ml_csv=ml_csv,
        out_dir=out_dir
    )
