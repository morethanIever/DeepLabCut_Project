# projects/projects.py
import os
import json
import shutil
from typing import List, Dict, Any
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
    }
    with open(_profile_path(projects_root, project_name), "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    return pdir

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

def set_dlc_config(projects_root: str, project_name: str, config_path: str) -> str:
    if not config_path:
        raise ValueError("config_path is empty.")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml not found: {config_path}")

    prof = load_profile(projects_root, project_name)
    prof.setdefault("dlc", {})
    prof["dlc"]["config_path"] = os.path.abspath(config_path)
    _save_profile(projects_root, project_name, prof)
    return prof["dlc"]["config_path"]

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

    # Prefer original DLC project path if present in config.yaml
    config_path_to_use = dst
    try:
        if yaml is None:
            raise RuntimeError("PyYAML not available")
        raw = bytes(data)
        cfg = yaml.safe_load(raw) if raw else {}
        project_path = cfg.get("project_path") if isinstance(cfg, dict) else None
        if isinstance(project_path, list):
            project_path = " ".join([str(x) for x in project_path if x is not None])
        if isinstance(project_path, str):
            project_path = " ".join(project_path.split())
            if project_path:
                candidate = os.path.join(project_path, "config.yaml")
                if os.path.exists(candidate):
                    config_path_to_use = os.path.abspath(candidate)
    except Exception:
        config_path_to_use = dst

    # profile update
    prof = load_profile(projects_root, project_name)
    prof.setdefault("dlc", {})
    prof["dlc"]["config_path"] = os.path.abspath(config_path_to_use)
    _save_profile(projects_root, project_name, prof)

    return prof["dlc"]["config_path"]
