import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
#import jobis

def run_clustering(feature_csv_list: str, out_dir: str, n_clusters=7):
    os.makedirs(out_dir, exist_ok=True)
    
    all_data_list = []
    video_indices = []
    print(f"[ML] Processing {len(feature_csv_list)} videos for global clustering...")
    
    for path in feature_csv_list:
        df = pd.read_csv(path)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 비디오별로 RobustScaler 적용 (개체차/카메라높이차 보정)
        scaler = RobustScaler()
        scaled_values = scaler.fit_transform(df)
        
        # 임시 데이터프레임 생성
        scaled_df = pd.DataFrame(scaled_values, columns=df.columns)
        all_data_list.append(scaled_df)
        video_indices.append(len(df))

    # 2. 모든 데이터 통합
    X_combined = pd.concat(all_data_list, axis=0).reset_index(drop=True)

    # 3. 통합 모델 학습 (Global KMeans)
    print(f"[ML] Training global model on {len(X_combined)} frames...")
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    global_labels = kmeans.fit_predict(X_combined)


    # 2) PCA (시각화용)
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X_combined)

    # 5. 결과를 다시 비디오별로 쪼개서 저장
    start_idx = 0
    for i, path in enumerate(feature_csv_list):
        end_idx = start_idx + video_indices[i]
        
        video_labels = global_labels[start_idx:end_idx]
        video_pc1 = Xp[start_idx:end_idx, 0]
        video_pc2 = Xp[start_idx:end_idx, 1]
        
        # 결과 저장
        out_df = pd.DataFrame({
            "pc1": video_pc1,
            "pc2": video_pc2,
            "cluster": video_labels
        })
        
        base = os.path.basename(path).replace("_features.csv", "")
        out_path = os.path.join(out_dir, f"{base}_global_clusters.csv")
        out_df.to_csv(out_path, index=False)
        
        print(f"  - Saved clusters for: {base}")
        start_idx = end_idx

        # 6. 통합 클러스터 프로필 (해석용) 저장
    # 원본 수치 기준으로 각 클러스터가 어떤 행동인지 확인하기 위해 원본 데이터와 결합
    # (주의: 원본 데이터를 다시 합쳐야 함)
    # 여기서는 간단히 마지막 비디오의 프로필 혹은 통합 프로필의 경향성을 확인
    print("\n[ML] Global clustering completed.")
    return out_path

if __name__ == "__main__":
    import sys

    # 인자가 최소 3개(파일들..., 출력디렉토리) 필요합니다.
    if len(sys.argv) < 3:
        print("Usage: python -m pipeline.ML.ml_cluster <feature_csv1> <feature_csv2> ... <out_dir>")
        sys.exit(1)

    # 마지막 인자를 out_dir로 설정하고, 그 앞의 모든 인자를 리스트로 묶어 csv_list로 만듭니다.
    feature_csv_list = sys.argv[1:-1]
    out_dir = sys.argv[-1]

    # 함수 호출 시 '==' 대신 '='를 사용하거나 순서대로 인자를 넣습니다.
    run_clustering(
        feature_csv_list=feature_csv_list,
        out_dir=out_dir
    )
