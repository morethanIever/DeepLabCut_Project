import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import hdbscan
import plotly.express as px


def apply_umap(df):
    import umap  # avoids importing parametric_umap / tensorflow
    reducer = umap(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    
    # 1. Select only numeric data
    X = df.select_dtypes(include=[np.number])
    # 2. Critical: Drop columns that are all zeros or NaNs (like start-of-video lags)
    X = X.loc[:, (X != 0).any(axis=0)].dropna(axis=1, how='all')
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 3. CONVERT TO NUMPY to avoid the "(slice(None, None, None), 0)" error
    X_values = X.values 
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_values)
    
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.01, random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    
    # Return as a simple DataFrame with the original index
    return pd.DataFrame(embedding, columns=['umap_1', 'umap_2'], index=df.index)

def save_umap_plot(df, video_path, output_dir="outputs/plots/ml", color_col=None):
    os.makedirs(output_dir, exist_ok=True)
    video_name = Path(video_path).stem
    
    # --- FIX: Check if UMAP was already computed in ml_features.py ---
    if 'umap_1' in df.columns and 'umap_2' in df.columns:
        print(f"[ML] UMAP already exists for {video_name}. Skipping re-computation.")
        # Extract the coordinates for KMeans if needed
        embedding = df[['umap_1', 'umap_2']].values
        df_plot = df.copy()
    else:
        print(f"[ML] Computing UMAP for {video_name}...")
        # This returns a DataFrame with 'umap_1' and 'umap_2'
        umap_df = apply_umap(df) 
        embedding = umap_df.values
        df_plot = pd.concat([df, umap_df], axis=1)

    # --- FIX: Use visual clustering only if no real behavior labels exist ---
    if color_col is None or color_col not in df_plot.columns:
        # Use embedding (the numpy array) to avoid indexing errors
        clusterer = KMeans(n_clusters=8, n_init=10, random_state=0)
        clusters = clusterer.fit_predict(embedding)
        df_plot = df.copy()
        df_plot['visual_cluster'] = clusters
        color_col = 'visual_cluster'
    
    # 3. Create the Plotly Figure
    fig = px.scatter(
        df_plot, 
        x='umap_1', 
        y='umap_2', 
        color=color_col,
        title=f"Behavioral Map Discovery: {video_name}",
        template="plotly_white",
        # Adding index to hover so you can find frame numbers
        hover_data={'umap_1':False, 'umap_2':False, 'speed':True} 
    )
    
    fig.update_traces(marker=dict(size=3, opacity=0.5))
    
    out_path = os.path.join(output_dir, f"{video_name}_umap.html")
    fig.write_html(out_path)
    
    return out_path, clusters