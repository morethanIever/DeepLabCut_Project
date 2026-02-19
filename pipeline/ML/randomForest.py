# pipeline/ML/randomForest.py
import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, List

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

DEFAULT_BEHAVIORS = ["sniffing", "grooming", "rearing", "turning", "moving", "rest", "fast_moving", "other"]


def _detect_label_format(df: pd.DataFrame, behaviors: List[str]):
    onehot_ok = all(b in df.columns for b in behaviors)
    if onehot_ok:
        onehot = df[behaviors].to_numpy()
        row_sum = onehot.sum(axis=1)

        # unlabeled rows -> NaN marker
        y = np.full(len(df), fill_value=-1, dtype=int)
        labeled = row_sum > 0
        y[labeled] = np.argmax(onehot[labeled], axis=1).astype(int)
        return y, behaviors

    if "behavior_label" in df.columns:
        labels = df["behavior_label"].astype(str)
        y = np.full(len(df), fill_value=-1, dtype=int)
        for i, s in enumerate(labels):
            if s in behaviors:
                y[i] = behaviors.index(s)
            elif s and s.lower() != "none":
                # unknown label -> other if exists
                y[i] = behaviors.index("other") if "other" in behaviors else -1
        return y, behaviors

    # more helpful error message
    cols_preview = ", ".join(list(df.columns)[:30])
    raise RuntimeError(
        "No labels found in --data CSV.\n"
        "You must train on a MERGED csv that includes either:\n"
        "  (A) onehot columns: " + ", ".join(behaviors) + "\n"
        "  (B) a 'behavior_label' column\n"
        f"Current CSV columns (first 30): {cols_preview}\n"
        "Fix: run merge_labels_into_features first, then pass that merged csv to --data."
    )


def _make_time_split(indices: np.ndarray, val_ratio: float):
    n = len(indices)
    n_val = max(1, int(round(n * val_ratio)))
    split = n - n_val
    return indices[:split], indices[split:]

def _class_counts(y: np.ndarray, class_names: List[str]) -> dict:
    counts = {}
    for i, name in enumerate(class_names):
        counts[name] = int((y == i).sum())
    return counts


def train_random_forest(
    merged_csv: str,
    out_dir: str,
    *,
    behaviors: Optional[List[str]] = None,
    val_ratio: float = 0.2,
    use_time_split: bool = True,
    use_train_mask: bool = True,     # ✅ boundary 제외 마스크 사용
    force_stratified_when_missing: bool = True,
    stratify_random_split: bool = True,
    n_estimators: int = 600,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 2,
    n_jobs: int = -1,
    random_state: int = 42,
):
    if behaviors is None:
        behaviors = DEFAULT_BEHAVIORS

    df = pd.read_csv(merged_csv)

    # y: -1 means unlabeled
    y, class_names = _detect_label_format(df, behaviors)

    # ---- training mask ----
    mask = np.ones(len(df), dtype=bool)

    # exclude unlabeled
    mask &= (y >= 0)

    # exclude boundary frames if provided
    if use_train_mask and "train_mask" in df.columns:
        mask &= (df["train_mask"].fillna(0).astype(int) == 1)

    # apply mask
    df_m = df.loc[mask].reset_index(drop=True)
    y_m = y[mask]

    if len(df_m) < 50:
        raise RuntimeError(f"Too few labeled frames after masking: {len(df_m)}. Check your annotations/train_mask.")

    counts_after_mask = _class_counts(y_m, class_names)
    print("[RF] Label counts after mask:", counts_after_mask)

    # ---- features ----
    drop_cols = set(["frame"])
    drop_cols.update([b for b in behaviors if b in df_m.columns])
    if "behavior_label" in df_m.columns:
        drop_cols.add("behavior_label")
    if "train_mask" in df_m.columns:
        drop_cols.add("train_mask")

    X = df_m.drop(columns=[c for c in drop_cols if c in df_m.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number]).fillna(0.0)

    if X.shape[1] == 0:
        raise RuntimeError("No numeric feature columns found after dropping labels/masks.")

    # ---- split indices ----
    idx = np.arange(len(df_m))
    if use_time_split:
        tr_idx, va_idx = _make_time_split(idx, val_ratio)
    else:
        if stratify_random_split:
            tr_idx, va_idx = train_test_split(
                idx, test_size=val_ratio, random_state=random_state, stratify=y_m
            )
        else:
            tr_idx, va_idx = train_test_split(
                idx, test_size=val_ratio, random_state=random_state, stratify=None
            )

    if use_time_split and force_stratified_when_missing:
        val_counts = _class_counts(y_m[va_idx], class_names)
        missing_in_val = [k for k, v in val_counts.items() if v == 0 and counts_after_mask.get(k, 0) > 0]
        if missing_in_val:
            print("[RF][WARN] Time split left some classes out of validation:", missing_in_val)
            try:
                tr_idx, va_idx = train_test_split(
                    idx, test_size=val_ratio, random_state=random_state, stratify=y_m
                )
                print("[RF][INFO] Switched to stratified random split to include all classes in validation.")
            except Exception as e:
                print("[RF][WARN] Stratified split failed; using random split without stratify. Error:", e)
                tr_idx, va_idx = train_test_split(
                    idx, test_size=val_ratio, random_state=random_state, stratify=None
                )

    X_train, y_train = X.iloc[tr_idx], y_m[tr_idx]
    X_val, y_val = X.iloc[va_idx], y_m[va_idx]

    # ---- class weights ----
    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=random_state,
        class_weight=class_weight,
        bootstrap=True,
    )

    clf.fit(X_train, y_train)

    pred = clf.predict(X_val)

    report = classification_report(
        y_val, pred, labels=np.arange(len(class_names)), target_names=class_names, zero_division=0
    )
    cm = confusion_matrix(y_val, pred, labels=np.arange(len(class_names)))

    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "rf_model.joblib")
    meta_path = os.path.join(out_dir, "rf_meta.joblib")
    report_path = os.path.join(out_dir, "val_report.txt")
    cm_path = os.path.join(out_dir, "val_confusion_matrix.csv")
    fi_path = os.path.join(out_dir, "feature_importance.csv")

    joblib.dump(clf, model_path)
    joblib.dump(
        {
            "class_names": class_names,
            "feature_columns": list(X.columns),
            "use_time_split": use_time_split,
            "val_ratio": val_ratio,
            "class_weight": class_weight,
            "params": clf.get_params(),
            "used_train_mask": (use_train_mask and "train_mask" in df.columns),
            "n_samples_total": len(df),
            "n_samples_used": len(df_m),
        },
        meta_path,
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(cm_path)

    fi = pd.DataFrame({"feature": X.columns, "importance": clf.feature_importances_})
    fi.sort_values("importance", ascending=False).to_csv(fi_path, index=False)

    print("[RF] Model:", model_path)
    print("[RF] Used samples:", len(df_m), "/", len(df))
    print("[RF] Report:\n", report)
    print("[RF] Confusion matrix saved:", cm_path)
    print("[RF] Feature importance saved:", fi_path)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="MERGED csv (features + labels + optional train_mask)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--time_split", action="store_true")
    p.add_argument("--random_split", action="store_true")
    p.add_argument("--no_train_mask", action="store_true", help="ignore train_mask column if present")
    p.add_argument("--no_force_stratified", action="store_true", help="do not fallback to stratified random split when time split misses classes")
    p.add_argument("--no_stratify_random_split", action="store_true", help="disable stratify for random split")
    p.add_argument("--n_estimators", type=int, default=600)
    p.add_argument("--max_depth", type=int, default=0, help="0 means None")
    p.add_argument("--min_leaf", type=int, default=2)
    args = p.parse_args()

    use_time = True
    if args.random_split:
        use_time = False
    if args.time_split:
        use_time = True

    md = None if args.max_depth == 0 else args.max_depth

    train_random_forest(
        args.data,
        args.out,
        val_ratio=args.val_ratio,
        use_time_split=use_time,
        use_train_mask=not args.no_train_mask,
        force_stratified_when_missing=not args.no_force_stratified,
        stratify_random_split=not args.no_stratify_random_split,
        n_estimators=args.n_estimators,
        max_depth=md,
        min_samples_leaf=args.min_leaf,
    )
