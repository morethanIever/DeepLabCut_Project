import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation

def animate_trajectory_mp4(kin_csv: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    video_stem = os.path.splitext(os.path.basename(kin_csv))[0]
    out_path = os.path.join(out_dir, f"{video_stem}_trajectory.mp4")
    if os.path.exists(out_path):
        print(f"Trajectory animation already exists, skipping: {out_path}")
        return out_path

    # 1. 데이터 로드
    df = pd.read_csv(kin_csv)
    x = df["spine_x"].to_numpy()
    y = df["spine_y"].to_numpy()
    speed = df["speed_px_s"].to_numpy()

    # 2. 그래프 초기 설정
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(x.min() - 20, x.max() + 20)
    ax.set_ylim(y.min() - 20, y.max() + 20)
    ax.invert_yaxis()  # 이미지 좌표계 반영
    ax.set_aspect('equal')
    ax.set_title("Rodent Trajectory Animation")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.grid(alpha=0.3)

    # 선 세그먼트를 담을 컬렉션 초기화
    lc = LineCollection([], cmap="inferno", norm=plt.Normalize(speed.min(), speed.max()))
    lc.set_linewidth(2)
    ax.add_collection(lc)

    # 현재 위치를 표시할 점
    current_point, = ax.plot([], [], 'ko', markersize=5) 
    
    # 컬러바 추가
    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label("Speed (px/s)")

    # 3. 애니메이션 업데이트 함수
    # step을 조절하여 영상의 속도와 프레임 수를 조절할 수 있습니다.
    step = 5 
    frames_idx = range(0, len(df), step)

    def update(frame_idx):
        if frame_idx < 2:
            return lc, current_point

        # 현재 프레임까지의 경로 추출
        curr_x = x[:frame_idx]
        curr_y = y[:frame_idx]
        curr_speed = speed[:frame_idx]

        # 경로를 선분(segments)으로 변환
        points = np.array([curr_x, curr_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # 컬렉션 업데이트
        lc.set_segments(segments)
        lc.set_array(curr_speed[:-1])
        
        # 현재 위치 점 업데이트
        current_point.set_data([curr_x[-1]], [curr_y[-1]])

        return lc, current_point

    # 4. 애니메이션 생성 및 저장
    print("애니메이션 생성 중... 데이터 양에 따라 시간이 걸릴 수 있습니다.")
    ani = FuncAnimation(fig, update, frames=frames_idx, blit=True, interval=30)

    # fps: 초당 프레임 수 (원본 영상의 FPS에 맞춰 조절하세요)
    ani.save(out_path, writer='ffmpeg', fps=30, dpi=150)
    plt.close()
    
    print(f"저장 완료: {out_path}")
    return out_path

