import os
import joblib
import numpy as np
import pandas as pd


def _entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def predict_rf_to_csv(
    ml_features_csv: str,
    model_dir: str,
    out_dir: str,
    *,
    top_k: int = 500,
    conf_thresh: float = 0.55,
    margin_thresh: float = 0.15,
    clip_len: int = 30,
):
    """
    Inputs:
      - ml_features_csv: features for ONE video (no labels needed)
      - model_dir: contains rf_model.joblib + rf_meta.joblib
    Outputs:
      - predictions_full.csv : frame-wise predicted label + metrics
      - outlier_candidates.csv : top-k uncertain/outlier frames + suggested clip start/end
    """
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "rf_model.joblib")
    meta_path = os.path.join(model_dir, "rf_meta.joblib")
    clf = joblib.load(model_path)
    meta = joblib.load(meta_path)

    class_names = meta["class_names"]
    feature_cols = meta["feature_columns"]

    df = pd.read_csv(ml_features_csv)

    # ensure frame exists
    if "frame" not in df.columns:
        df.insert(0, "frame", np.arange(len(df), dtype=int))

    # align features to training columns (missing -> 0)
    X = df.reindex(columns=feature_cols, fill_value=0.0)
    X = X.select_dtypes(include=[np.number]).fillna(0.0)

    probs = clf.predict_proba(X.to_numpy())
    pred_idx = np.argmax(probs, axis=1)
    pred_label = [class_names[i] for i in pred_idx]

    # confidence, margin, entropy
    p_sorted = np.sort(probs, axis=1)
    p1 = p_sorted[:, -1]
    p2 = p_sorted[:, -2] if probs.shape[1] > 1 else np.zeros_like(p1)
    margin = p1 - p2
    ent = _entropy(probs)

    out = pd.DataFrame({
        "frame": df["frame"].astype(int),
        "pred_label": pred_label,
        "conf": p1,
        "margin": margin,
        "entropy": ent,
    })

    out["is_uncertain"] = (out["conf"] < conf_thresh) | (out["margin"] < margin_thresh)

    # save full predictions
    full_path = os.path.join(out_dir, "predictions_full.csv")
    out.to_csv(full_path, index=False)

    # select candidates: prioritize uncertain + highest entropy
    cand = out.sort_values(["is_uncertain", "entropy"], ascending=[False, False]).head(top_k).copy()

    # Add suggested clip start/end to match your annotator (clip-based)
    half = clip_len // 2
    cand["clip_start"] = (cand["frame"] - half).clip(lower=0).astype(int)
    cand["clip_end"] = (cand["clip_start"] + clip_len - 1).astype(int)

    cand_path = os.path.join(out_dir, "outlier_candidates.csv")
    cand.to_csv(cand_path, index=False)

    print("[PRED] Full predictions saved:", full_path)
    print("[PRED] Outlier candidates saved:", cand_path)
    return full_path, cand_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, help="ml_features.csv for one video")
    p.add_argument("--model_dir", required=True, help="directory containing rf_model.joblib + rf_meta.joblib")
    p.add_argument("--out_dir", required=True, help="where to save prediction CSVs")
    p.add_argument("--top_k", type=int, default=500)
    p.add_argument("--conf_thresh", type=float, default=0.55)
    p.add_argument("--margin_thresh", type=float, default=0.15)
    p.add_argument("--clip_len", type=int, default=30)
    args = p.parse_args()

    predict_rf_to_csv(
        args.features,
        args.model_dir,
        args.out_dir,
        top_k=args.top_k,
        conf_thresh=args.conf_thresh,
        margin_thresh=args.margin_thresh,
        clip_len=args.clip_len,
    )
