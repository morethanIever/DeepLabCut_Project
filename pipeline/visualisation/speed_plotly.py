import pandas as pd
import plotly.express as px
import os

def animate_speed_plotly(kin_csv: str, out_dir: str):
    # 1. 데이터 로드
    df = pd.read_csv(kin_csv)

    # Plotly 애니메이션을 위해 누적 데이터를 생성 (선이 점점 길어지게 보이기 위함)
    # 데이터가 너무 많을 경우 성능을 위해 [::5] 등으로 샘플링할 수 있습니다.
    df_anim = pd.concat([df.iloc[:i] for i in range(1, len(df) + 1, 5)]) # 5프레임씩 건너뜀
    
    # 애니메이션 효과를 위해 각 시점별 데이터 세트에 그룹 식별자 추가
    # Note: 단순히 시점만 보여주려면 아래 그래프에서 animation_frame='frame'만 써도 됩니다.

    # 2. 그래프 생성
    fig = px.line(
        df, 
        x="frame", 
        y="speed_px_s",
        title="Rodent Speed Over Time (Plotly)",
        labels={"frame": "Frame", "speed_px_s": "Speed (px/s)"},
        template="plotly_white"
    )

    # 3. 애니메이션 추가 (실시간 추적 효과)
    # 프레임별로 데이터가 쌓이는 방식을 위해 원본 데이터 사용
    fig = px.scatter(
        df, 
        x="frame", 
        y="speed_px_s",
        animation_frame="frame", # 아래 슬라이더를 움직여 시간에 따른 변화 확인
        range_x=[df['frame'].min(), df['frame'].max()],
        range_y=[df['speed_px_s'].min(), df['speed_px_s'].max() * 1.1],
        title="Rodent Speed Animation"
    )

    # 선 그래프 추가
    fig.add_scatter(x=df["frame"], y=df["speed_px_s"], mode="lines", line=dict(color="rgba(0,0,255,0.2)"), name="Full Path")

    # 4. 저장 및 출력
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "speed_animation.html")
    
    # HTML로 저장하면 브라우저에서 인터랙티브하게 조작 가능합니다.
    fig.write_html(out_path)
    fig.show() # 실행 시 브라우저에서 즉시 열림

    return out_path

