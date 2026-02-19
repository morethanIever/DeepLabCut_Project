# projects/projects.py
import os
import json
import shutil
import configparser
import glob
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
try:
    import yaml
except Exception:  # pragma: no cover - optional dependency in some envs
    yaml = None

PROFILE_NAME = "profile.json"

def _project_dir(projects_root: str, project_name: str) -> str:
    return os.path.join(projects_root, project_name)

def _profile_path(projects_root: str, project_name: str) -> str:
    return os.path.join(_project_dir(projects_root, project_name), PROFILE_NAME)

def list_projects(projects_root: str) -> List[str]:
    if not os.path.exists(projects_root):
        return []
    out = []
    for name in os.listdir(projects_root):
        p = os.path.join(projects_root, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, PROFILE_NAME)):
            out.append(name)
    out.sort()
    return out

def create_project(projects_root: str, project_name: str) -> str:
    if not project_name:
        raise ValueError("Project name is empty.")
    os.makedirs(projects_root, exist_ok=True)

    pdir = _project_dir(projects_root, project_name)
    if os.path.exists(pdir):
        raise ValueError(f"Project already exists: {project_name}")

    os.makedirs(pdir, exist_ok=True)
    os.makedirs(os.path.join(pdir, "assets"), exist_ok=True)
    os.makedirs(os.path.join(pdir, "outputs"), exist_ok=True)

    profile = {
        "name": project_name,
        "dlc": {"config_path": ""},
        "simba": {"config_path": ""},
    }
    with open(_profile_path(projects_root, project_name), "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    return pdir

def create_simba_config_template(projects_root: str, project_name: str) -> str:
    """
    Create a minimal SimBA project_config.ini template in the project assets dir
    and update profile.simba.config_path to point to it.
    """
    proj_dir = _project_dir(projects_root, project_name)
    if not os.path.exists(proj_dir):
        raise FileNotFoundError(f"Project dir not found: {proj_dir}")

    assets_dir = os.path.join(proj_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    cfg_path = os.path.join(assets_dir, "simba_project_config.ini")
    if not os.path.exists(cfg_path):
        template = (
            "[General settings]\n"
            "project_path = \n"
            "project_name = \n"
            "workflow_file_type = csv\n"
            "animal_no = 1\n"
            "os_system = Windows\n"
            "\n"
            "[SML settings]\n"
            "model_dir = \n"
            "model_path_1 = \n"
            "model_path_2 = \n"
            "model_path_3 = \n"
            "model_path_4 = \n"
            "model_path_5 = \n"
            "model_path_6 = \n"
            "no_targets = 0\n"
            "target_name_1 = \n"
            "target_name_2 = \n"
            "target_name_3 = \n"
            "target_name_4 = \n"
            "target_name_5 = \n"
            "target_name_6 = \n"
            "\n"
            "[threshold_settings]\n"
            "threshold_1 = None\n"
            "threshold_2 = None\n"
            "threshold_3 = None\n"
            "threshold_4 = None\n"
            "threshold_5 = None\n"
            "threshold_6 = None\n"
            "bp_threshold_sklearn = 0.0\n"
            "\n"
            "[Minimum_bout_lengths]\n"
            "min_bout_1 = None\n"
            "min_bout_2 = None\n"
            "min_bout_3 = None\n"
            "min_bout_4 = None\n"
            "min_bout_5 = None\n"
            "min_bout_6 = None\n"
            "\n"
            "[Frame settings]\n"
            "distance_mm = 3\n"
            "\n"
            "[Line plot settings]\n"
            "\n"
            "[Path plot settings]\n"
            "\n"
            "[ROI settings]\n"
            "\n"
            "[Directionality settings]\n"
            "\n"
            "[process movements]\n"
            "\n"
            "[create ensemble settings]\n"
            "pose_estimation_body_parts = \n"
            "classifier = None\n"
            "train_test_size = 0.2\n"
            "model_to_run = RF\n"
            "train_test_split_type = FRAMES\n"
            "under_sample_setting = None\n"
            "under_sample_ratio = None\n"
            "over_sample_setting = None\n"
            "over_sample_ratio = None\n"
            "rf_n_estimators = 2000\n"
            "rf_min_sample_leaf = 1\n"
            "rf_max_features = sqrt\n"
            "rf_n_jobs = -1\n"
            "rf_criterion = entropy\n"
            "rf_max_depth = None\n"
            "generate_rf_model_meta_data_file = None\n"
            "generate_example_decision_tree = None\n"
            "generate_example_decision_tree_fancy = None\n"
            "generate_features_importance_log = None\n"
            "generate_features_importance_bar_graph = None\n"
            "compute_feature_permutation_importance = None\n"
            "generate_sklearn_learning_curves = None\n"
            "generate_precision_recall_curves = None\n"
            "generate_classification_report = None\n"
            "compute_shap_scores = None\n"
            "compute_partial_dependency = None\n"
            "class_weights = None\n"
            "class_custom_weights = None\n"
            "n_feature_importance_bars = None\n"
            "learning_curve_k_splits = None\n"
            "learningcurve_shuffle_data_splits = None\n"
            "shap_target_present_cnt = 0\n"
            "shap_target_absent_cnt = 0\n"
            "shap_save_iter = None\n"
            "shap_multiprocess = False\n"
            "cuda = False\n"
            "\n"
            "[Multi animal IDs]\n"
            "id_list = 1\n"
            "\n"
            "[Outlier settings]\n"
            "movement_criterion = None\n"
            "location_criterion = None\n"
        )
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(template)

    prof = load_profile(projects_root, project_name)
    prof.setdefault("simba", {})
    prof["simba"]["config_path"] = os.path.abspath(cfg_path)
    _save_profile(projects_root, project_name, prof)
    return prof["simba"]["config_path"]

def autofill_simba_config(projects_root: str, project_name: str) -> str:
    """
    Auto-fill SimBA config with project-specific paths and create needed folders.
    """
    prof = load_profile(projects_root, project_name)
    cfg_path = prof.get("simba", {}).get("config_path", "")
    if not cfg_path or not os.path.exists(cfg_path):
        raise FileNotFoundError("SimBA config not found. Create or set the path first.")

    proj_dir = _project_dir(projects_root, project_name)
    simba_root = os.path.join(proj_dir, "simba")
    project_folder = os.path.join(simba_root, "project_folder")
    videos_dir = os.path.join(proj_dir, "outputs", "videos")
    pose_dir = os.path.join(proj_dir, "outputs", "poses")
    model_dir = os.path.join(simba_root, "models")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    cfg = configparser.ConfigParser()
    cfg.read(cfg_path, encoding="utf-8")
    for section in [
        "General settings",
        "SML settings",
        "threshold_settings",
        "Minimum_bout_lengths",
        "Frame settings",
        "Line plot settings",
        "Path plot settings",
        "ROI settings",
        "Directionality settings",
        "process movements",
        "create ensemble settings",
        "Multi animal IDs",
        "Outlier settings",
        "Project",
        "Videos",
        "Pose",
        "Models",
    ]:
        if section not in cfg:
            cfg[section] = {}

    cfg["General settings"]["project_name"] = project_name
    cfg["General settings"]["project_path"] = os.path.abspath(project_folder)
    cfg["General settings"].setdefault("workflow_file_type", "csv")
    cfg["General settings"].setdefault("animal_no", "1")
    cfg["General settings"].setdefault("os_system", "Windows")
    cfg["SML settings"].setdefault("model_dir", os.path.abspath(model_dir))
    cfg["SML settings"].setdefault("no_targets", "0")
    cfg["Multi animal IDs"].setdefault("id_list", "1")

    # Try infer bodyparts from latest DLC pose CSV (if present)
    pose_candidates = glob.glob(os.path.join(proj_dir, "outputs", "poses", "*_filtered.csv"))
    bodyparts = []
    if pose_candidates:
        pose_candidates.sort(key=os.path.getmtime, reverse=True)
        try:
            df = pd.read_csv(pose_candidates[0], header=[0, 1, 2])
            bps = sorted(set(df.columns.get_level_values(1)))
            bodyparts = [bp for bp in bps if isinstance(bp, str) and bp.lower() not in {"bodyparts", "coords", "scorer"}]
        except Exception:
            bodyparts = []
    if bodyparts:
        # SimBA only supports specific counts; otherwise use user_defined
        count = str(len(bodyparts))
        if count in {"4", "7", "8", "9", "14", "16"}:
            cfg["create ensemble settings"]["pose_estimation_body_parts"] = count
        else:
            cfg["create ensemble settings"]["pose_estimation_body_parts"] = "user_defined"
    else:
        cfg["create ensemble settings"].setdefault("pose_estimation_body_parts", "")
    cfg["Project"]["project_name"] = project_name
    cfg["Project"]["project_path"] = os.path.abspath(project_folder)
    cfg["Videos"]["video_dir"] = os.path.abspath(videos_dir)
    cfg["Videos"].setdefault("px_per_mm", "30.0")
    cfg["Videos"].setdefault("fps", "30")
    cfg["Pose"]["pose_format"] = "DLC"
    cfg["Pose"]["pose_dir"] = os.path.abspath(pose_dir)
    cfg["Models"]["classifier_dir"] = os.path.abspath(model_dir)

    with open(cfg_path, "w", encoding="utf-8") as f:
        cfg.write(f)

    return os.path.abspath(cfg_path)


def update_simba_model_settings(
    projects_root: str,
    project_name: str,
    *,
    model_dir: str,
    target_names: List[str],
    threshold: float = 0.5,
    min_bout: int = 5,
) -> str:
    """
    Update SimBA model settings in project_config.ini.
    """
    prof = load_profile(projects_root, project_name)
    cfg_path = prof.get("simba", {}).get("config_path", "")
    if not cfg_path or not os.path.exists(cfg_path):
        raise FileNotFoundError("SimBA config not found. Create or set the path first.")

    cfg = configparser.ConfigParser()
    cfg.read(cfg_path, encoding="utf-8")
    if "SML settings" not in cfg:
        cfg["SML settings"] = {}
    if "threshold_settings" not in cfg:
        cfg["threshold_settings"] = {}
    if "Minimum_bout_lengths" not in cfg:
        cfg["Minimum_bout_lengths"] = {}

    cfg["SML settings"]["model_dir"] = os.path.abspath(model_dir)
    cfg["SML settings"]["no_targets"] = str(len(target_names))

    for i, name in enumerate(target_names, start=1):
        safe_name = str(name).strip()
        cfg["SML settings"][f"target_name_{i}"] = safe_name
        cfg["SML settings"][f"model_path_{i}"] = os.path.abspath(
            os.path.join(model_dir, f"{safe_name}.sav")
        )
        cfg["threshold_settings"][f"threshold_{i}"] = str(threshold)
        cfg["Minimum_bout_lengths"][f"min_bout_{i}"] = str(min_bout)

    with open(cfg_path, "w", encoding="utf-8") as f:
        cfg.write(f)

    return os.path.abspath(cfg_path)


def set_simba_labels(projects_root: str, project_name: str, labels: List[str]) -> List[str]:
    prof = load_profile(projects_root, project_name)
    prof.setdefault("simba", {})
    prof["simba"]["labels"] = labels
    _save_profile(projects_root, project_name, prof)
    return labels


def get_simba_labels(projects_root: str, project_name: str) -> List[str]:
    prof = load_profile(projects_root, project_name)
    labels = prof.get("simba", {}).get("labels", [])
    if not isinstance(labels, list):
        return []
    return [str(x) for x in labels if str(x).strip()]

def load_profile(projects_root: str, project_name: str) -> Dict[str, Any]:
    p = _profile_path(projects_root, project_name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Profile not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_profile(projects_root: str, project_name: str, prof: Dict[str, Any]) -> None:
    p = _profile_path(projects_root, project_name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(prof, f, ensure_ascii=False, indent=2)

def _resolve_real_dlc_config_path(config_path: str, *, assets_dir: Optional[str] = None) -> str:
    """
    If config.yaml references a project_path, prefer the config at that location.
    Fall back to the given config_path when resolution fails.
    """
    try:
        if not config_path or not os.path.exists(config_path) or yaml is None:
            return config_path
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        proj_path = cfg.get("project_path")
        if isinstance(proj_path, list) and proj_path:
            proj_path = proj_path[0]
        if isinstance(proj_path, str):
            candidate = os.path.abspath(os.path.join(proj_path, "config.yaml"))
            if os.path.exists(candidate):
                if assets_dir and os.path.abspath(candidate).startswith(os.path.abspath(assets_dir) + os.sep):
                    return config_path
                return candidate
    except Exception:
        return config_path
    return config_path

def set_dlc_config(projects_root: str, project_name: str, config_path: str) -> str:
    if not config_path:
        raise ValueError("config_path is empty.")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml not found: {config_path}")

    prof = load_profile(projects_root, project_name)
    prof.setdefault("dlc", {})
    assets_dir = os.path.join(_project_dir(projects_root, project_name), "assets")
    real_path = _resolve_real_dlc_config_path(config_path, assets_dir=assets_dir)
    if os.path.abspath(real_path).startswith(os.path.abspath(assets_dir) + os.sep):
        raise ValueError(
            "DLC config path points to the project assets folder. "
            "Please provide the original DLC project config.yaml path instead."
        )
    prof["dlc"]["config_path"] = os.path.abspath(real_path)
    _save_profile(projects_root, project_name, prof)
    return prof["dlc"]["config_path"]

def get_dlc_reference(projects_root: str, project_name: str) -> Dict[str, Any]:
    prof = load_profile(projects_root, project_name)
    return prof.get("dlc", {}).get("reference", {}) or {}

def set_dlc_reference(projects_root: str, project_name: str, reference: Dict[str, Any]) -> Dict[str, Any]:
    prof = load_profile(projects_root, project_name)
    prof.setdefault("dlc", {})
    prof["dlc"]["reference"] = reference
    _save_profile(projects_root, project_name, prof)
    return prof["dlc"]["reference"]

def set_simba_config(projects_root: str, project_name: str, config_path: str) -> str:
    if not config_path:
        raise ValueError("config_path is empty.")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"SimBA project_config.ini not found: {config_path}")

    prof = load_profile(projects_root, project_name)
    prof.setdefault("simba", {})
    prof["simba"]["config_path"] = os.path.abspath(config_path)
    _save_profile(projects_root, project_name, prof)
    return prof["simba"]["config_path"]

def import_dlc_config_file(projects_root: str, project_name: str, uploaded_file) -> str:
    """
    Streamlit file_uploader로 받은 config.yaml(UploadedFile)을
    projects/<project>/assets/config.yaml 로 저장하고,
    profile.json의 dlc.config_path를 그 경로로 설정한다.
    """
    if uploaded_file is None:
        raise ValueError("uploaded_file is None.")

    proj_dir = _project_dir(projects_root, project_name)
    if not os.path.exists(proj_dir):
        raise FileNotFoundError(f"Project dir not found: {proj_dir}")

    assets_dir = os.path.join(proj_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    # 항상 같은 이름으로 저장(프로젝트당 1개 config를 기본으로)
    dst = os.path.abspath(os.path.join(assets_dir, "config.yaml"))

    # uploaded_file은 streamlit의 UploadedFile
    # .getbuffer() 또는 .read() 사용 가능
    data = uploaded_file.getbuffer()
    with open(dst, "wb") as f:
        f.write(data)

    # profile update: prefer real DLC project path if present in config.yaml
    prof = load_profile(projects_root, project_name)
    prof.setdefault("dlc", {})
    real_path = _resolve_real_dlc_config_path(dst, assets_dir=assets_dir)
    if os.path.abspath(real_path).startswith(os.path.abspath(assets_dir) + os.sep):
        raise ValueError(
            "Uploaded DLC config.yaml resolves to the project assets folder. "
            "Please use the Direct Path field to set the original DLC project config.yaml path."
        )
    prof["dlc"]["config_path"] = os.path.abspath(real_path)
    _save_profile(projects_root, project_name, prof)

    return prof["dlc"]["config_path"]

def set_project_settings(
    projects_root: str,
    project_name: str,
    *,
    experiment_type: Optional[str] = None,
    camera_view: Optional[str] = None,
) -> Dict[str, Any]:
    prof = load_profile(projects_root, project_name)
    prof.setdefault("settings", {})
    if experiment_type is not None:
        prof["settings"]["experiment_type"] = experiment_type
    if camera_view is not None:
        prof["settings"]["camera_view"] = camera_view
    _save_profile(projects_root, project_name, prof)
    return prof["settings"]

def delete_project(projects_root: str, project_name: str) -> None:
    if not project_name:
        raise ValueError("Project name is empty.")
    pdir = _project_dir(projects_root, project_name)
    if not os.path.exists(pdir):
        raise FileNotFoundError(f"Project dir not found: {pdir}")
    shutil.rmtree(pdir)
