import os
import configparser
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


def train_simba_models_from_annotations(
    config_path: str,
    *,
    annotations_csv: str,
    labels: List[str],
    video_stem: str,
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
    features_csv = os.path.join(project_path, "csv", "features_extracted", f"{video_stem}.csv")
    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"SimBA features_extracted not found: {features_csv}")

    targets_dir = os.path.join(project_path, "csv", "targets_inserted")
    targets_csv = os.path.join(targets_dir, f"{video_stem}.csv")

    ann_df = pd.read_csv(annotations_csv)
    _build_targets_inserted(features_csv, ann_df, labels, targets_csv)

    # Train one model per label
    from simba.model.train_rf import TrainRandomForestClassifier

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
        trainer = TrainRandomForestClassifier(config_path=config_path)
        trainer.run()
        trainer.save()
        model_path = os.path.join(model_dir, "generated_models", f"{lab}.sav")
        if os.path.exists(model_path):
            # copy to model_dir root for inference
            dst = os.path.join(model_dir, f"{lab}.sav")
            try:
                if os.path.abspath(model_path) != os.path.abspath(dst):
                    import shutil
                    shutil.copy(model_path, dst)
            except Exception:
                pass
            saved_models.append(dst)
    return saved_models
