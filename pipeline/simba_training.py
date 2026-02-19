import os
import configparser
import glob
import cv2
from typing import List

import pandas as pd


def _read_project_path(config_path: str) -> str:
    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")
    for section in ("General settings", "Project"):
        if cfg.has_section(section) and cfg.has_option(section, "project_path"):
            p = cfg.get(section, "project_path", fallback="").strip()
            if p:
                return os.path.abspath(p)
    return os.path.abspath(os.path.dirname(config_path))


def _read_pose_dir(config_path: str, project_path: str) -> str:
    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")
    for section in ("Pose", "Project"):
        if cfg.has_section(section) and cfg.has_option(section, "pose_dir"):
            p = cfg.get(section, "pose_dir", fallback="").strip()
            if p:
                return os.path.abspath(p)
    return os.path.join(project_path, "csv", "pose")


def _read_video_dir(config_path: str, project_path: str) -> str:
    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")
    if cfg.has_section("Videos") and cfg.has_option("Videos", "video_dir"):
        p = cfg.get("Videos", "video_dir", fallback="").strip()
        if p:
            return os.path.abspath(p)
    return os.path.join(project_path, "videos")


def _find_video_for_stem(video_dir: str, stem: str) -> str | None:
    exts = ["mp4", "avi", "mov", "mkv"]
    for ext in exts:
        cand = os.path.join(video_dir, f"{stem}.{ext}")
        if os.path.exists(cand):
            return cand
    # fallback: any file containing stem
    candidates = glob.glob(os.path.join(video_dir, f"*{stem}*.*"))
    if candidates:
        return max(candidates, key=os.path.getmtime)
    return None


def _get_video_props(video_path: str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, (w, h)


def _ensure_video_info_csv(
    config_path: str,
    project_path: str,
    video_stem: str,
    *,
    video_path: str | None = None,
) -> str | None:
    logs_dir = os.path.join(project_path, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    out_path = os.path.join(logs_dir, "video_info.csv")

    resolved_video = None
    if video_path and os.path.exists(video_path):
        resolved_video = video_path
    else:
        video_dir = _read_video_dir(config_path, project_path)
        resolved_video = _find_video_for_stem(video_dir, video_stem)
    if not resolved_video:
        return None

    fps, (w, h) = _get_video_props(resolved_video)
    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")
    px_per_mm = None
    distance_mm = None
    if cfg.has_section("Videos"):
        px_per_mm = cfg.get("Videos", "px_per_mm", fallback="").strip()
    if cfg.has_section("Frame settings"):
        distance_mm = cfg.get("Frame settings", "distance_mm", fallback="").strip()

    try:
        px_per_mm_val = float(px_per_mm) if px_per_mm not in (None, "", "None") else 0.0
    except Exception:
        px_per_mm_val = 0.0
    try:
        distance_mm_val = float(distance_mm) if distance_mm not in (None, "", "None") else 1.0
    except Exception:
        distance_mm_val = 1.0

    video_name = os.path.splitext(os.path.basename(resolved_video))[0]
    row = {
        "Video": video_name,
        "fps": float(fps),
        "Resolution_width": float(w),
        "Resolution_height": float(h),
        "Distance_in_mm": float(distance_mm_val),
        "pixels/mm": float(px_per_mm_val),
    }

    if os.path.exists(out_path):
        df = pd.read_csv(out_path)
        if "Video" in df.columns and (df["Video"].astype(str) == row["Video"]).any():
            return out_path
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(out_path, index=False)
    return out_path


def _ensure_project_bp_names(config_path: str, project_path: str) -> str | None:
    """
    Ensure SimBA project_bp_names.csv exists. Try to infer from a pose CSV.
    """
    logs_dir = os.path.join(project_path, "logs", "measures", "pose_configs", "bp_names")
    os.makedirs(logs_dir, exist_ok=True)
    out_path = os.path.join(logs_dir, "project_bp_names.csv")
    if os.path.exists(out_path):
        return out_path

    pose_dir = _read_pose_dir(config_path, project_path)
    candidates = glob.glob(os.path.join(pose_dir, "*.csv"))
    if not candidates:
        return None
    pose_csv = max(candidates, key=os.path.getmtime)

    try:
        df = pd.read_csv(pose_csv, header=[0, 1, 2])
        bodyparts = sorted(set(df.columns.get_level_values(1)))
        bodyparts = [
            bp for bp in bodyparts
            if isinstance(bp, str) and bp.lower() not in {"bodyparts", "coords", "scorer"}
        ]
    except Exception:
        bodyparts = []
    if not bodyparts:
        return None
    with open(out_path, "w", encoding="utf-8") as f:
        for bp in bodyparts:
            f.write(f"{bp}\n")
    return out_path


def _update_config_for_classifier(
    config_path: str,
    labels: List[str],
    model_dir: str,
    threshold: float,
    min_bout: int,
    clf_name: str,
):
    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")
    if "SML settings" not in cfg:
        cfg["SML settings"] = {}
    if "threshold_settings" not in cfg:
        cfg["threshold_settings"] = {}
    if "Minimum_bout_lengths" not in cfg:
        cfg["Minimum_bout_lengths"] = {}
    if "create ensemble settings" not in cfg:
        cfg["create ensemble settings"] = {}

    cfg["SML settings"]["model_dir"] = os.path.abspath(model_dir)
    cfg["SML settings"]["no_targets"] = str(len(labels))
    cfg["create ensemble settings"]["classifier"] = clf_name
    cfg["create ensemble settings"]["train_test_size"] = "0.2"
    cfg["create ensemble settings"]["model_to_run"] = "RF"
    cfg["create ensemble settings"]["train_test_split_type"] = "FRAMES"
    cfg["create ensemble settings"]["under_sample_setting"] = "None"
    cfg["create ensemble settings"]["under_sample_ratio"] = "None"
    cfg["create ensemble settings"]["over_sample_setting"] = "None"
    cfg["create ensemble settings"]["over_sample_ratio"] = "None"
    cfg["create ensemble settings"]["rf_n_estimators"] = "2000"
    cfg["create ensemble settings"]["rf_min_sample_leaf"] = "1"
    cfg["create ensemble settings"]["rf_max_features"] = "sqrt"
    cfg["create ensemble settings"]["rf_n_jobs"] = "-1"
    cfg["create ensemble settings"]["rf_criterion"] = "entropy"
    cfg["create ensemble settings"]["rf_max_depth"] = "None"
    cfg["create ensemble settings"]["generate_rf_model_meta_data_file"] = "None"
    cfg["create ensemble settings"]["generate_example_decision_tree"] = "None"
    cfg["create ensemble settings"]["generate_example_decision_tree_fancy"] = "None"
    cfg["create ensemble settings"]["generate_features_importance_log"] = "None"
    cfg["create ensemble settings"]["generate_features_importance_bar_graph"] = "None"
    cfg["create ensemble settings"]["compute_feature_permutation_importance"] = "None"
    cfg["create ensemble settings"]["generate_sklearn_learning_curves"] = "None"
    cfg["create ensemble settings"]["generate_precision_recall_curves"] = "None"
    cfg["create ensemble settings"]["generate_classification_report"] = "None"
    cfg["create ensemble settings"]["compute_shap_scores"] = "None"
    cfg["create ensemble settings"]["compute_partial_dependency"] = "None"
    cfg["create ensemble settings"]["class_weights"] = "None"
    cfg["create ensemble settings"]["class_custom_weights"] = "None"
    cfg["create ensemble settings"]["n_feature_importance_bars"] = "None"
    cfg["create ensemble settings"]["learning_curve_k_splits"] = "None"
    cfg["create ensemble settings"]["learningcurve_shuffle_data_splits"] = "None"
    cfg["create ensemble settings"]["shap_target_present_cnt"] = "0"
    cfg["create ensemble settings"]["shap_target_absent_cnt"] = "0"
    cfg["create ensemble settings"]["shap_save_iter"] = "None"
    cfg["create ensemble settings"]["shap_multiprocess"] = "False"
    cfg["create ensemble settings"]["cuda"] = "False"

    for i, name in enumerate(labels, start=1):
        cfg["SML settings"][f"target_name_{i}"] = name
        cfg["SML settings"][f"model_path_{i}"] = os.path.abspath(os.path.join(model_dir, f"{name}.sav"))
        cfg["threshold_settings"][f"threshold_{i}"] = str(threshold)
        cfg["Minimum_bout_lengths"][f"min_bout_{i}"] = str(min_bout)

    with open(config_path, "w", encoding="utf-8") as f:
        cfg.write(f)


def _build_targets_inserted(
    features_csv: str,
    annotations_df: pd.DataFrame,
    labels: List[str],
    out_csv: str,
):
    feats = pd.read_csv(features_csv)
    n = len(feats)
    if "frame" not in feats.columns:
        feats.insert(0, "frame", range(n))
    else:
        feats = feats.sort_values("frame").reset_index(drop=True)

    for lab in labels:
        if lab not in feats.columns:
            feats[lab] = 0

    for _, row in annotations_df.iterrows():
        lab = str(row.get("label", "")).strip()
        if lab not in labels:
            continue
        s = int(row.get("start_frame", 0))
        e = int(row.get("end_frame", 0))
        if e < s:
            s, e = e, s
        s = max(0, s)
        e = min(n - 1, e)
        feats.loc[(feats["frame"] >= s) & (feats["frame"] <= e), lab] = 1

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    feats.to_csv(out_csv, index=False)


def _build_features_from_merged(
    merged_csv: str,
    labels: List[str],
    out_csv: str,
) -> None:
    df = pd.read_csv(merged_csv)
    if "frame" not in df.columns:
        df.insert(0, "frame", range(len(df)))
    drop_cols = set(labels)
    drop_cols.update(["behavior_label", "train_mask"])
    keep = [c for c in df.columns if c not in drop_cols]
    feats = df[keep].copy()
    # Keep only numeric columns + frame to match SimBA features_extracted expectations
    numeric_cols = feats.select_dtypes(include=["number"]).columns.tolist()
    if "frame" not in numeric_cols:
        numeric_cols = ["frame"] + [c for c in numeric_cols if c != "frame"]
    feats = feats[numeric_cols]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    feats.to_csv(out_csv, index=False)


def _ensure_features_extracted(
    config_path: str,
    project_path: str,
    video_stem: str,
    *,
    labels: List[str],
    merged_csv_fallback: str | None = None,
    allow_fallback: bool = False,
) -> str:
    """
    Ensure SimBA features_extracted CSV exists. Try SimBA feature extraction first,
    then fall back to building from merged CSV if provided.
    """
    features_csv = os.path.join(project_path, "csv", "features_extracted", f"{video_stem}.csv")
    if os.path.exists(features_csv):
        return features_csv

    # Try SimBA feature extraction if available
    try:
        from simba.utils.cli.cli_tools import feature_extraction_runner
        feature_extraction_runner(config_path=config_path)
    except Exception:
        pass

    if os.path.exists(features_csv):
        return features_csv

    if allow_fallback and merged_csv_fallback and os.path.exists(merged_csv_fallback):
        _build_features_from_merged(merged_csv_fallback, labels, features_csv)
        return features_csv

    raise FileNotFoundError(f"SimBA features_extracted not found: {features_csv}")


def train_simba_models_from_annotations(
    config_path: str,
    *,
    annotations_csv: str,
    labels: List[str],
    video_stem: str,
    video_path: str | None = None,
    merged_csv_fallback: str | None = None,
    allow_feature_fallback: bool = False,
    threshold: float = 0.5,
    min_bout: int = 5,
) -> List[str]:
    """
    Train one binary SimBA RF model per label using annotations and features_extracted.
    Returns list of saved model paths.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"SimBA config not found: {config_path}")
    if not os.path.exists(annotations_csv):
        raise FileNotFoundError(f"Annotations CSV not found: {annotations_csv}")
    if not labels:
        raise ValueError("No labels provided.")

    project_path = _read_project_path(config_path)
    bp_path = _ensure_project_bp_names(config_path, project_path)
    if not bp_path:
        logs_dir = os.path.join(project_path, "logs", "measures", "pose_configs", "bp_names")
        expected = os.path.join(logs_dir, "project_bp_names.csv")
        raise FileNotFoundError(
            "SimBA project_bp_names.csv is missing and could not be inferred. "
            f"Expected: {expected}. "
            "Fix: set Pose.pose_dir in the SimBA config to a valid DLC pose CSV folder "
            "and try again."
        )
    vi_path = _ensure_video_info_csv(config_path, project_path, video_stem, video_path=video_path)
    if not vi_path:
        logs_dir = os.path.join(project_path, "logs")
        expected = os.path.join(logs_dir, "video_info.csv")
        raise FileNotFoundError(
            "SimBA video_info.csv is missing and could not be inferred. "
            f"Expected: {expected}. "
            "Fix: set Videos.video_dir in the SimBA config to a valid video folder "
            "and try again."
        )
    features_csv = _ensure_features_extracted(
        config_path,
        project_path,
        video_stem,
        labels=labels,
        merged_csv_fallback=merged_csv_fallback,
        allow_fallback=allow_feature_fallback,
    )
    # Drop any unnamed index columns to keep feature counts stable
    try:
        feats_df = pd.read_csv(features_csv)
        unnamed = [c for c in feats_df.columns if c.lower().startswith("unnamed")]
        if unnamed:
            feats_df = feats_df.drop(columns=unnamed)
            feats_df.to_csv(features_csv, index=False)
    except Exception:
        pass

    targets_dir = os.path.join(project_path, "csv", "targets_inserted")
    targets_csv = os.path.join(targets_dir, f"{video_stem}.csv")

    ann_df = pd.read_csv(annotations_csv)
    _build_targets_inserted(features_csv, ann_df, labels, targets_csv)

    # Train one model per label
    from simba.model.train_rf import TrainRandomForestClassifier

    # Point SimBA at the base models dir; SimBA will create a single generated_models folder inside.
    model_dir = os.path.join(os.path.dirname(project_path), "models")
    os.makedirs(model_dir, exist_ok=True)
    saved_models = []
    for lab in labels:
        _update_config_for_classifier(
            config_path=config_path,
            labels=labels,
            model_dir=model_dir,
            threshold=threshold,
            min_bout=min_bout,
            clf_name=lab,
        )
        # Remove any stale model to avoid feature mismatch from older trainings
        stale_model = os.path.join(model_dir, "generated_models", f"{lab}.sav")
        try:
            if os.path.exists(stale_model):
                os.remove(stale_model)
        except Exception:
            pass
        trainer = TrainRandomForestClassifier(config_path=config_path)
        trainer.run()
        trainer.save()
        model_path = os.path.join(model_dir, "generated_models", f"{lab}.sav")
        if os.path.exists(model_path):
            saved_models.append(model_path)
    return saved_models


def train_simba_models_from_annotations_multi(
    config_path: str,
    *,
    annotations_csvs: List[str],
    labels: List[str],
    video_stems: List[str],
    video_paths: List[str] | None = None,
    merged_csv_fallback: str | None = None,
    allow_feature_fallback: bool = False,
    threshold: float = 0.5,
    min_bout: int = 5,
) -> List[str]:
    """
    Train SimBA models across multiple videos. For each video:
    - ensure features_extracted exists
    - build targets_inserted using that video's annotations
    Then train one RF per label (SimBA uses all targets_inserted in the project).
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"SimBA config not found: {config_path}")
    if not annotations_csvs:
        raise ValueError("No annotations CSVs provided.")
    if not labels:
        raise ValueError("No labels provided.")
    if len(video_stems) != len(annotations_csvs):
        raise ValueError("video_stems length must match annotations_csvs length.")
    if video_paths and len(video_paths) != len(video_stems):
        raise ValueError("video_paths length must match video_stems length.")

    project_path = _read_project_path(config_path)
    bp_path = _ensure_project_bp_names(config_path, project_path)
    if not bp_path:
        logs_dir = os.path.join(project_path, "logs", "measures", "pose_configs", "bp_names")
        expected = os.path.join(logs_dir, "project_bp_names.csv")
        raise FileNotFoundError(
            "SimBA project_bp_names.csv is missing and could not be inferred. "
            f"Expected: {expected}. "
            "Fix: set Pose.pose_dir in the SimBA config to a valid DLC pose CSV folder "
            "and try again."
        )

    for idx, ann_path in enumerate(annotations_csvs):
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Annotations CSV not found: {ann_path}")
        v_stem = video_stems[idx]
        v_path = video_paths[idx] if video_paths else None
        vi_path = _ensure_video_info_csv(config_path, project_path, v_stem, video_path=v_path)
        if not vi_path:
            logs_dir = os.path.join(project_path, "logs")
            expected = os.path.join(logs_dir, "video_info.csv")
            raise FileNotFoundError(
                "SimBA video_info.csv is missing and could not be inferred. "
                f"Expected: {expected}. "
                "Fix: set Videos.video_dir in the SimBA config to a valid video folder "
                "and try again."
            )
        features_csv = _ensure_features_extracted(
            config_path,
            project_path,
            v_stem,
            labels=labels,
            merged_csv_fallback=merged_csv_fallback,
            allow_fallback=allow_feature_fallback,
        )
        # Drop any unnamed index columns to keep feature counts stable
        try:
            feats_df = pd.read_csv(features_csv)
            unnamed = [c for c in feats_df.columns if c.lower().startswith("unnamed")]
            if unnamed:
                feats_df = feats_df.drop(columns=unnamed)
                feats_df.to_csv(features_csv, index=False)
        except Exception:
            pass

        targets_dir = os.path.join(project_path, "csv", "targets_inserted")
        targets_csv = os.path.join(targets_dir, f"{v_stem}.csv")
        ann_df = pd.read_csv(ann_path)
        _build_targets_inserted(features_csv, ann_df, labels, targets_csv)

    # Train one model per label (SimBA uses all targets_inserted in the project)
    from simba.model.train_rf import TrainRandomForestClassifier

    # Point SimBA at the base models dir; SimBA will create a single generated_models folder inside.
    model_dir = os.path.join(os.path.dirname(project_path), "models")
    os.makedirs(model_dir, exist_ok=True)
    saved_models = []
    for lab in labels:
        _update_config_for_classifier(
            config_path=config_path,
            labels=labels,
            model_dir=model_dir,
            threshold=threshold,
            min_bout=min_bout,
            clf_name=lab,
        )
        stale_model = os.path.join(model_dir, "generated_models", f"{lab}.sav")
        try:
            if os.path.exists(stale_model):
                os.remove(stale_model)
        except Exception:
            pass
        trainer = TrainRandomForestClassifier(config_path=config_path)
        trainer.run()
        trainer.save()
        model_path = os.path.join(model_dir, "generated_models", f"{lab}.sav")
        if os.path.exists(model_path):
            saved_models.append(model_path)
    return saved_models
