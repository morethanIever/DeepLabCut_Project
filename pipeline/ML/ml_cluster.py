import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def run_clustering(ml_csv: str, out_dir: str, n_clusters=5):
    df = pd.read_csv(ml_csv)

    X = df.values

    # 1) normalize
    Xs = StandardScaler().fit_transform(X)

    # 2) PCA (시각화용)
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(Xs)

    # 3) clustering
    km = KMeans(n_clusters=n_clusters, random_state=0)
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
