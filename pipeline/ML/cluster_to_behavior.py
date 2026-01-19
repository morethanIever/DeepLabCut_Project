import pandas as pd
import numpy as np
import os

def map_cluster_using_rule_behavior(
    ml_clusters_csv: str,
    rule_behavior_csv: str,
    out_dir="outputs/ml"
):
    os.makedirs(out_dir, exist_ok=True)

    clusters = pd.read_csv(ml_clusters_csv)
    beh = pd.read_csv(rule_behavior_csv)

    # frame index alignment check
    assert len(clusters) == len(beh), "Frame length mismatch"

    df = pd.DataFrame({
        "cluster": clusters["cluster"],
        "behavior": beh["behavior"],
        "confidence": beh.get("confidence", 1.0),
    })

    # ------------------------------------
    # Cluster × Behavior count table
    # ------------------------------------
    count_table = (
        df
        .groupby(["cluster", "behavior"])
        .size()
        .unstack(fill_value=0)
    )

    count_path = os.path.join(out_dir, "cluster_behavior_counts.csv")
    count_table.to_csv(count_path)

    # ------------------------------------
    # Dominant behavior per cluster
    # ------------------------------------
    behavior_map = {}

    for cluster_id, row in count_table.iterrows():
        dominant_behavior = row.idxmax()
        behavior_map[cluster_id] = dominant_behavior

    map_df = pd.DataFrame({
        "cluster": list(behavior_map.keys()),
        "mapped_behavior": list(behavior_map.values())
    })

    map_path = os.path.join(out_dir, "cluster_behavior_map.csv")
    map_df.to_csv(map_path, index=False)

    # ------------------------------------
    # Apply mapping to frames
    # ------------------------------------
    clusters["behavior_from_cluster"] = clusters["cluster"].map(behavior_map)

    frame_path = os.path.join(out_dir, "behavior_from_cluster.csv")
    clusters.to_csv(frame_path, index=False)

    print("[OK] Cluster → behavior mapping completed")
    print(map_df)

    return map_path, frame_path

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print(
            "Usage:\n"
            "python -m pipeline.ML.cluster_to_behavior "
            "<ml_clusters.csv> <rule_behavior.csv>"
        )
        sys.exit(1)

    ml_clusters_csv = sys.argv[1]
    rule_behavior_csv = sys.argv[2]

    map_cluster_using_rule_behavior(
        ml_clusters_csv=ml_clusters_csv,
        rule_behavior_csv=rule_behavior_csv,
    )