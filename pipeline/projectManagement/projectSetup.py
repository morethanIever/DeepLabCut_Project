import os
import configparser
import re
from pathlib import Path
import streamlit as st

from projects.projects import (
    list_projects, create_project, load_profile, set_dlc_config, import_dlc_config_file, delete_project, set_project_settings, set_simba_config, create_simba_config_template, autofill_simba_config, update_simba_model_settings, set_simba_labels, get_simba_labels
)
from pipeline.projectManagement.init_session import reset_analysis_state

PROJECTS_ROOT = "projects"

def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

def render_project_setup_page():
    with st.container(border=True):
        st.title("Project & Config Setup")
        st.caption("Work top-to-bottom. Nothing is saved until you click a button.")

        projects = list_projects(PROJECTS_ROOT)
        has_projects = bool(projects)

        if st.session_state.pop("clear_new_project_name", False):
            st.session_state.new_project_name = ""

    with st.container(border=True):
        st.subheader("1) Project")
        st.caption("Create, switch, or delete projects.")

        selected_proj = st.selectbox(
            "Existing Projects",
            options=projects if has_projects else ["(no projects yet)"],
            index=projects.index(st.session_state.active_project) if st.session_state.active_project in projects else 0,
            disabled=not has_projects,
            key="project_select",
        )

        new_name = st.text_input("New Project Name", key="new_project_name").strip()

        if st.button("Create Project", type="primary") and new_name:
            create_project(PROJECTS_ROOT, new_name)
            st.session_state.active_project = new_name
            reset_analysis_state()
            st.session_state.clear_new_project_name = True
            _rerun()


        if has_projects and selected_proj != st.session_state.active_project:
            st.session_state.active_project = selected_proj
            reset_analysis_state()
            _rerun()

        
        confirm_delete = st.checkbox("I understand this will permanently delete the project.")
        if st.button("Delete Project", disabled=not confirm_delete):
            delete_project(PROJECTS_ROOT, st.session_state.active_project)
            st.session_state.active_project = None
            reset_analysis_state()
            _rerun()

        with st.container(border=True):
            st.subheader("2) DLC Config")
            st.caption("Upload or paste the path to your DeepLabCut config.yaml.")

            prof = load_profile(PROJECTS_ROOT, st.session_state.active_project)
            st.session_state.active_profile = prof

            cfg_up = st.file_uploader("Upload config.yaml", type=["yaml", "yml"])
            if st.button("Apply Uploaded Config", type="primary") and cfg_up:
                import_dlc_config_file(PROJECTS_ROOT, st.session_state.active_project, cfg_up)
                _rerun()

            cur_cfg = prof.get("dlc", {}).get("config_path", "")
            new_cfg_path = st.text_input("Direct Path", value=cur_cfg)
            if st.button("Update Path", type="primary"):
                set_dlc_config(PROJECTS_ROOT, st.session_state.active_project, new_cfg_path)
                _rerun()

            st.caption(f"Status: {'Connected' if os.path.exists(cur_cfg) else 'Not Found'}")

        with st.container(border=True):
            st.subheader("3) SimBA Config")
            st.caption("Set the SimBA config path and edit settings.")

            prof = load_profile(PROJECTS_ROOT, st.session_state.active_project)
            cur_cfg = prof.get("simba", {}).get("config_path", "")
            new_cfg_path = st.text_input("SimBA project_config.ini Path", value=cur_cfg)
            

            st.caption(f"Status: {'Connected' if os.path.exists(cur_cfg) else 'Not Found'}")

            with st.expander("Quick Actions", expanded=True):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button("Create SimBA Config Template", type="primary"):
                        create_simba_config_template(PROJECTS_ROOT, st.session_state.active_project)
                        _rerun()
                with col_b:
                    if st.button("Auto-fill SimBA Config", type="primary"):
                        try:
                            autofill_simba_config(PROJECTS_ROOT, st.session_state.active_project)
                            st.success("Auto-filled SimBA config.")
                            _rerun()
                        except Exception as e:
                            st.error(f"Auto-fill failed: {e}")
                with col_c:
                    if st.button("Update SimBA Path", type="primary"):
                        if not new_cfg_path.strip():
                            st.warning("Please enter a SimBA config path before clicking Update.")
                        else:
                            set_simba_config(PROJECTS_ROOT, st.session_state.active_project, new_cfg_path)
                            _rerun()

            cfg_path = cur_cfg if os.path.exists(cur_cfg) else None
            if cfg_path:
                with st.expander("Interactive SimBA Config Editor", expanded=False):
                    cfg = configparser.ConfigParser()
                    cfg.read(cfg_path, encoding="utf-8")

                    def _ensure_section(name: str) -> None:
                        if name not in cfg:
                            cfg[name] = {}

                    def _get(section: str, key: str, default: str = "") -> str:
                        return cfg.get(section, key, fallback=default)

                    for section in [
                        "General settings",
                        "SML settings",
                        "threshold_settings",
                        "Minimum_bout_lengths",
                        "Pose",
                        "Videos",
                        "Project",
                    ]:
                        _ensure_section(section)

                    st.caption("Edit core SimBA settings with form fields. Use the raw editor below for advanced options.")

                    col1, col2 = st.columns(2)
                    with col1:
                        project_name = st.text_input(
                            "Project name",
                            value=_get("General settings", "project_name"),
                            key="simba_cfg_project_name",
                        )
                        project_path = st.text_input(
                            "Project path",
                            value=_get("General settings", "project_path"),
                            key="simba_cfg_project_path",
                        )
                        workflow_file_type = st.text_input(
                            "Workflow file type",
                            value=_get("General settings", "workflow_file_type", "csv"),
                            key="simba_cfg_workflow_file_type",
                        )
                        animal_no_raw = _get("General settings", "animal_no", "1")
                        try:
                            animal_no_val = int(animal_no_raw)
                        except Exception:
                            animal_no_val = 1
                        animal_no = st.number_input(
                            "Animal count",
                            min_value=1,
                            value=animal_no_val,
                            step=1,
                            key="simba_cfg_animal_no",
                        )
                    with col2:
                        os_system = st.selectbox(
                            "OS system",
                            ["Windows", "Linux", "Mac"],
                            index=["Windows", "Linux", "Mac"].index(_get("General settings", "os_system", "Windows"))
                            if _get("General settings", "os_system", "Windows") in ["Windows", "Linux", "Mac"] else 0,
                            key="simba_cfg_os_system",
                        )
                        model_dir = st.text_input(
                            "Model directory",
                            value=_get("SML settings", "model_dir"),
                            key="simba_cfg_model_dir",
                        )
                        pose_dir = st.text_input(
                            "Pose directory",
                            value=_get("Pose", "pose_dir"),
                            key="simba_cfg_pose_dir",
                        )
                        video_dir = st.text_input(
                            "Video directory",
                            value=_get("Videos", "video_dir"),
                            key="simba_cfg_video_dir",
                        )

                    st.markdown("**Targets, thresholds, and min bout lengths**")
                    no_targets_raw = _get("SML settings", "no_targets", "0")
                    try:
                        no_targets_val = int(no_targets_raw)
                    except Exception:
                        no_targets_val = 0
                    no_targets = st.number_input(
                        "Number of target behaviors",
                        min_value=0,
                        max_value=20,
                        value=no_targets_val,
                        step=1,
                        key="simba_cfg_no_targets",
                    )

                    target_names = []
                    model_paths = []
                    thresholds = []
                    min_bouts = []
                    for i in range(1, int(no_targets) + 1):
                        col_a, col_b, col_c, col_d = st.columns([2, 3, 2, 2])
                        with col_a:
                            target_names.append(
                                st.text_input(
                                    f"Target {i} name",
                                    value=_get("SML settings", f"target_name_{i}"),
                                    key=f"simba_cfg_target_name_{i}",
                                )
                            )
                        with col_b:
                            model_paths.append(
                                st.text_input(
                                    f"Target {i} model path",
                                    value=_get("SML settings", f"model_path_{i}"),
                                    key=f"simba_cfg_model_path_{i}",
                                )
                            )
                        with col_c:
                            thresholds.append(
                                st.text_input(
                                    f"Target {i} threshold",
                                    value=_get("threshold_settings", f"threshold_{i}", "None"),
                                    key=f"simba_cfg_threshold_{i}",
                                )
                            )
                        with col_d:
                            min_bouts.append(
                                st.text_input(
                                    f"Target {i} min bout",
                                    value=_get("Minimum_bout_lengths", f"min_bout_{i}", "None"),
                                    key=f"simba_cfg_min_bout_{i}",
                                )
                            )

                    if st.button("Save Interactive SimBA Config"):
                        cfg["General settings"]["project_name"] = project_name
                        cfg["General settings"]["project_path"] = project_path
                        cfg["General settings"]["workflow_file_type"] = workflow_file_type
                        cfg["General settings"]["animal_no"] = str(int(animal_no))
                        cfg["General settings"]["os_system"] = os_system
                        cfg["SML settings"]["model_dir"] = model_dir
                        cfg["SML settings"]["no_targets"] = str(int(no_targets))
                        cfg["Pose"]["pose_dir"] = pose_dir
                        cfg["Videos"]["video_dir"] = video_dir
                        cfg["Project"]["project_name"] = project_name
                        cfg["Project"]["project_path"] = project_path

                        max_idx = 0
                        pattern = re.compile(r"_(\d+)$")
                        for section in ["SML settings", "threshold_settings", "Minimum_bout_lengths"]:
                            for key in cfg[section].keys():
                                m = pattern.search(key)
                                if m:
                                    try:
                                        max_idx = max(max_idx, int(m.group(1)))
                                    except Exception:
                                        pass
                        max_idx = max(max_idx, int(no_targets))

                        for i in range(1, int(no_targets) + 1):
                            name = target_names[i - 1].strip()
                            cfg["SML settings"][f"target_name_{i}"] = name
                            model_path = model_paths[i - 1].strip()
                            if not model_path and name and model_dir:
                                model_path = os.path.abspath(os.path.join(model_dir, f"{name}.sav"))
                            cfg["SML settings"][f"model_path_{i}"] = model_path

                            thr = thresholds[i - 1].strip()
                            cfg["threshold_settings"][f"threshold_{i}"] = thr if thr else "None"
                            mb = min_bouts[i - 1].strip()
                            cfg["Minimum_bout_lengths"][f"min_bout_{i}"] = mb if mb else "None"

                        for i in range(int(no_targets) + 1, max_idx + 1):
                            for section, prefix in [
                                ("SML settings", "target_name_"),
                                ("SML settings", "model_path_"),
                                ("threshold_settings", "threshold_"),
                                ("Minimum_bout_lengths", "min_bout_"),
                            ]:
                                key = f"{prefix}{i}"
                                if key in cfg[section]:
                                    del cfg[section][key]

                        with open(cfg_path, "w", encoding="utf-8") as f:
                            cfg.write(f)
                        st.success("Saved interactive SimBA config.")
                        _rerun()
                        
                st.write("---")
                with st.expander("**Raw SimBA config editor**", expanded=False):
                    try:
                        raw_cfg = Path(cfg_path).read_text(encoding="utf-8")
                    except Exception:
                        raw_cfg = ""
                    cfg_text = st.text_area(
                        "SimBA project_config.ini",
                        value=raw_cfg,
                        height=300,
                    )
            if st.button("Save SimBA Config", type="primary"):
                try:
                    Path(cfg_path).write_text(cfg_text, encoding="utf-8")
                    st.success("Saved SimBA config.")
                    _rerun()
                except Exception as e:
                    st.error(f"Failed to save: {e}")
            else:
                st.info("Create or set a SimBA config path to edit it here.")

            st.subheader("**SimBA Model Setup (in-app)**")
            st.caption("Use this if you have trained SimBA models and want to set targets/thresholds.")
            current_labels = get_simba_labels(PROJECTS_ROOT, st.session_state.active_project)
            labels_raw = st.text_area(
                "Project behavior labels (comma-separated)",
                value=", ".join(current_labels),
                height=80,
                key="simba_labels_input"
            )
            if st.button("Save Project Labels", type="primary"):
                labels = [t.strip() for t in labels_raw.split(",") if t.strip()]
                set_simba_labels(PROJECTS_ROOT, st.session_state.active_project, labels)
                st.success("Saved project labels.")
                _rerun()

            model_dir = st.text_input(
                "SimBA model folder",
                value=prof.get("simba", {}).get("model_dir", os.path.join(PROJECTS_ROOT, st.session_state.active_project, "simba", "models")),
                key="simba_model_dir_input"
            )
            targets_raw = st.text_area(
                "Target behaviors (comma-separated)",
                value="",
                height=80,
                key="simba_targets_input"
            )
            threshold = st.number_input("Default threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
            min_bout = st.number_input("Default min bout (frames)", min_value=0, value=5, step=1)

            if st.button("Apply SimBA Model Settings", type="primary"):
                targets = [t.strip() for t in targets_raw.split(",") if t.strip()]
                if not targets:
                    st.error("Please enter at least one behavior name.")
                else:
                    try:
                        update_simba_model_settings(
                            PROJECTS_ROOT,
                            st.session_state.active_project,
                            model_dir=model_dir,
                            target_names=targets,
                            threshold=threshold,
                            min_bout=min_bout,
                        )
                        st.success("SimBA model settings saved.")
                        _rerun()
                    except Exception as e:
                        st.error(f"Failed to save model settings: {e}")

        # Experiment type & camera settings (per-project)
        with st.container(border=True):
            settings = prof.get("settings", {})
            st.subheader("4) Experiment & Camera")
            st.caption("Saved per-project.")
            st.markdown("**Experiment Type**")
            exp_default = settings.get("experiment_type", "Open Field")
            st.session_state.experiment_type = st.selectbox(
                "Select Experiment Type",
                ["Open Field", "NOP", "VR", "Maze"],
                index=["Open Field", "NOP", "VR", "Maze"].index(exp_default)
                if exp_default in ["Open Field", "NOP", "VR", "Maze"] else 0,
            )

            st.markdown("**Camera Setting**")
            cam_default = settings.get("camera_view", "Top View")
            st.session_state.camera_view = st.selectbox(
                "Select Camera View",
                ["Top View", "Side", "Down"],
                index=["Top View", "Side", "Down"].index(cam_default)
                if cam_default in ["Top View", "Side", "Down"] else 0,
            )

            settings_saved = False
            if (
                st.session_state.experiment_type != exp_default
                or st.session_state.camera_view != cam_default
            ):
                set_project_settings(
                    PROJECTS_ROOT,
                    st.session_state.active_project,
                    experiment_type=st.session_state.experiment_type,
                    camera_view=st.session_state.camera_view,
                )
                


            st.button("Save the Setting", type="primary")
            settings_saved = True
            if settings_saved:
                st.caption("Saved")



        if st.button("Go to Main Page", type="primary"):
            st.session_state.page = "main"
            _rerun()
