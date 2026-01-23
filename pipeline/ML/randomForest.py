import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def train_behavior_model(data_path=r"outputs/ml/master_training_set.csv"):
    # 1. Load the unified data
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run your preparation script first.")
        return
    
    df = pd.read_csv(data_path)
    
    # 2. Define Features (X) and Target (y)
    # We exclude metadata and UMAP coordinates because they are video-specific
    non_feature_cols = [
        'visual_cluster', 'umap_1', 'umap_2', 
        'behavior_name', 'frame', 'video_id'
    ]
    
    # Select only the kinematic features and their lags
    X = df.drop(columns=[c for c in non_feature_cols if c in df.columns])
    y = df['behavior_name']
    
    # 3. Split into Train and Test sets for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    
    # 4. Initialize and train the Random Forest
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=20, 
        n_jobs=-1, 
        random_state=42,
        class_weight='balanced'  # Important if some behaviors occur less often
    )
    
    rf.fit(X_train, y_train)
    
    # 5. Evaluate the model
    y_pred = rf.predict(X_test)
    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, y_pred))
    
    # 6. Save the trained model and the feature list
    os.makedirs("models", exist_ok=True)
    model_data = {
        'model': rf,
        'features': X.columns.tolist() # Saving feature names is critical for inference
    }
    joblib.dump(model_data, "models/global_behavior_rf.pkl")
    print("\n[OK] Model saved to models/global_behavior_rf.pkl")

if __name__ == "__main__":
    train_behavior_model()