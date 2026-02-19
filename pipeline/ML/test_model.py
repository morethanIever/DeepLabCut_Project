import pandas as pd
import joblib
import os
import sys

def run_inference(kin_csv, model_path="models/global_behavior_rf.pkl"):
    # 1. Load the Model Data
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return
    
    saved_data = joblib.load(model_path)
    rf_model = saved_data['model']
    required_features = saved_data['features']
    
    # 2. Load and Prepare the Data
    # Note: This assumes you are passing the ml_features.csv 
    # which already has the 10 lags and smoothed features.
    df = pd.read_csv(kin_csv)
    
    # Ensure we only use the features the model was trained on
    # and maintain the correct column order
    X = df[required_features]
    
    # 3. Predict Behaviors
    print(f"Running inference on {len(df)} frames...")
    predictions = rf_model.predict(X)
    probabilities = rf_model.predict_proba(X).max(axis=1) # Confidence scores
    
    # 4. Save Results
    df['predicted_behavior'] = predictions
    df['prediction_confidence'] = probabilities
    
    output_path = kin_csv.replace(".csv", "_predicted.csv")
    df.to_csv(output_path, index=False)
    
    print(f"\n[OK] Predictions saved to: {output_path}")
    print("\nBehavior Distribution in this video:")
    print(df['predicted_behavior'].value_counts())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <path_to_ml_features_csv>")
    else:
        run_inference(sys.argv[1])