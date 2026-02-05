import os
import streamlit as st

# 가상의 라이브러리 (사용자 환경에 맞춰 임포트)
from projects.projects import (
    list_projects, create_project, load_profile, set_dlc_config, import_dlc_config_file, delete_project, set_project_settings
)
from pipeline.projectManagement.init_session import reset_analysis_state

PROJECTS_ROOT = "projects"

def render_project_setup_page():
    st.title("Project & Config Setup")

    projects = list_projects(PROJECTS_ROOT)
    has_projects = bool(projects)

    if st.session_state.pop("clear_new_project_name", False):
        st.session_state.new_project_name = ""

    with st.container(border=True):
        st.subheader("Select or Create Project")

        selected_proj = st.selectbox(
            "Existing Projects",
            options=projects if has_projects else ["(no projects yet)"],
            index=projects.index(st.session_state.active_project) if st.session_state.active_project in projects else 0,
            disabled=not has_projects,
            key="project_select",
        )

        new_name = st.text_input("New Project Name", key="new_project_name").strip()

        if st.button("Create Project", use_container_width=True) and new_name:
            create_project(PROJECTS_ROOT, new_name)
            st.session_state.active_project = new_name
            reset_analysis_state()
            st.session_state.clear_new_project_name = True
            st.rerun()


        if has_projects and selected_proj != st.session_state.active_project:
            st.session_state.active_project = selected_proj
            reset_analysis_state()
            st.rerun()

        
        confirm_delete = st.checkbox("I understand this will permanently delete the project.")
        if st.button("Delete Project", use_container_width=True, disabled=not confirm_delete):
            delete_project(PROJECTS_ROOT, st.session_state.active_project)
            st.session_state.active_project = None
            reset_analysis_state()
            st.rerun()

        with st.container(border=True):
            st.subheader("DLC Config")

            prof = load_profile(PROJECTS_ROOT, st.session_state.active_project)
            st.session_state.active_profile = prof

            cfg_up = st.file_uploader("Upload config.yaml", type=["yaml", "yml"])
            if st.button("Apply Uploaded Config", use_container_width=True) and cfg_up:
                import_dlc_config_file(PROJECTS_ROOT, st.session_state.active_project, cfg_up)
                st.rerun()

            cur_cfg = prof.get("dlc", {}).get("config_path", "")
            new_cfg_path = st.text_input("Direct Path", value=cur_cfg)
            if st.button("Update Path", use_container_width=True):
                set_dlc_config(PROJECTS_ROOT, st.session_state.active_project, new_cfg_path)
                st.rerun()

            st.caption(f"Status: {'Connected' if os.path.exists(cur_cfg) else 'Not Found'}")

        # Experiment type & camera settings (per-project)
        with st.container(border=True):
            settings = prof.get("settings", {})
            st.subheader("Experiment Type")
            exp_default = settings.get("experiment_type", "Open Field")
            st.session_state.experiment_type = st.selectbox(
                "Select Experiment Type",
                ["Open Field", "NOP", "VR", "Maze"],
                index=["Open Field", "NOP", "VR", "Maze"].index(exp_default)
                if exp_default in ["Open Field", "NOP", "VR", "Maze"] else 0,
            )

            st.subheader("Camera Setting")
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
                


            st.button("Save the Setting", use_container_width=True)
            settings_saved = True
            if settings_saved:
                st.caption("Saved")

        if st.button("Go to Main Page", type="primary", use_container_width=True):
            st.session_state.page = "main"
            st.rerun()
