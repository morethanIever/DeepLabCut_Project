from pathlib import Path
import streamlit as st


from pipeline.projectManagement.init_session import init_session_state, reset_analysis_state, _get_latest_file
PROJECTS_ROOT = "projects"

def sync_project_data():
    """프로젝트 폴더 내의 결과물을 CSV 해시 파일명에 맞춰 정밀하게 동기화"""
    active = st.session_state.active_project
    if not active or active == "(no projects yet)": return

    base_dir = Path(PROJECTS_ROOT) / active / "outputs"
    plots_dir = base_dir / "plots"

    # 1. Kinematics CSV 찾기 (가장 최근 분석된 데이터 기준)
    if not st.session_state.kin_csv_path:
        st.session_state.kin_csv_path = _get_latest_file([base_dir / "**/*kin*.csv"])
    # 2. CSV가 있으면 해당 해시값(prefix)으로 시작하는 모든 PNG를 긁어옴
    if st.session_state.kin_csv_path and plots_dir.exists():
        csv_stem = Path(st.session_state.kin_csv_path).stem
        if csv_stem.endswith("_kinematics"):
            prefix = csv_stem[: -len("_kinematics")]
        else:
            prefix = csv_stem

        # 매핑 규칙 정의 (사용자 제공 파일명 기반)
        mapping = {
            "speed_plot": f"{prefix}_kinematics_speed.png",
            "trajectory_plot": f"{prefix}_kinematics_trajectory_speed.png", 
            "trajectory_behavior": f"{prefix}_kinematics_trajectory_behavior.png",
            "turning_rate_plot_path": f"{prefix}_kinematics_turning_rate.png",
            "trajectory_turning_plot": f"{prefix}_kinematics_trajectory_turning.png",
            "nop_plot": f"{prefix}_kinematics_nop.png"
        }

        for key, filename in mapping.items():
            full_path = plots_dir / filename
            found = None
            # NOP의 경우 하위 폴더에 있을 수 있으므로 체크
            if not full_path.exists():
                alt_path = base_dir / "nop" / filename
                if alt_path.exists():
                    full_path = alt_path

            # 파일이 실제로 존재하면 세션에 강제 업데이트
            if full_path.exists():
                st.session_state[key] = str(full_path)
            else:
                # 파일이 없으면 혹시 모르니 패턴 검색 시도
                found = _get_latest_file([plots_dir / f"{prefix}*{key.split('_')[0]}*.png"])
                if found:
                    st.session_state[key] = found

    # 3. SimBA results sync (if configured)
    prof = st.session_state.active_profile or {}
    simba_cfg = prof.get("simba", {}).get("config_path", "")
    if simba_cfg and Path(simba_cfg).exists():
        simba_proj = Path(simba_cfg).parent
        simba_csv = _get_latest_file([
            simba_proj / "csv" / "machine_results" / "*.csv",
            simba_proj / "csv" / "classifier_validation" / "*.csv",
            simba_proj / "csv" / "inference" / "*.csv",
            simba_proj / "csv" / "**" / "*.csv",
        ])
        if simba_csv:
            st.session_state.simba_machine_csv = str(simba_csv)
