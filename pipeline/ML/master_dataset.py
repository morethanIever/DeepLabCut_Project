import pandas as pd
import glob

def create_master_dataset():
    # Define your manual mappings for each video file
    # Ensure these names match exactly (e.g., "Walk" vs "Walking")
    configs = [
        {'path': r'outputs\ml\fad92c0398211bcb9c0ef182e0fe65cb_ml_features.csv', 'map': {0: 'Fast move', 1: 'Sniffing', 2: 'Move', 3: 'Rest', 4: 'Grooming', 5: 'Sniffing', 6: 'Turning', 7: 'Rearing'}},
        {'path': r'outputs\ml\bf57b37b35cd86006f143dbf36d41402_ml_features.csv', 'map': {0: 'Move', 1: 'Grooming', 2: 'Turning', 3: 'Rearing', 4: 'Move', 5: 'Sniffing', 6: 'Fast Move', 7: 'Rest'}},
        {'path': r'outputs\ml\e8c9ec435e1183ddfd78181e05ecfac1_ml_features.csv', 'map': {0: 'Rest', 1: 'Turning', 2: 'Sniffing', 3: 'Rearing', 4: 'Grooming', 5: 'Fast Move', 6: 'Move', 7: 'Fast Move'}},
        {'path': r'outputs\ml\0abdd7ba277898156608d9b842e97d68_ml_features.csv', 'map': {0: 'Sniffing', 1: 'Grooming', 2: 'Fast move', 3: 'Turning', 4: 'Rest', 5: 'Rearing', 6: 'Move', 7:'Grooming'}},
    ]
    
    all_dfs = []
    for cfg in configs:
        df = pd.read_csv(cfg['path'])
        # Translate numbers to names
        df['behavior_name'] = df['visual_cluster'].map(cfg['map'])
        # Drop rows that weren't mapped or are considered 'Noise'
        df = df.dropna(subset=['behavior_name'])
        all_dfs.append(df)
    
    # Merge all 4 videos into one master training set
    master_df = pd.concat(all_dfs, axis=0, ignore_index=True)
    master_df.to_csv("outputs/ml/master_training_set.csv", index=False)
    print("Master dataset created with", len(master_df), "total samples.")
    return master_df

if __name__ == "__main__":
    # Ensure the output directory exists
    import os
    if not os.path.exists("outputs/ml"):
        os.makedirs("outputs/ml")
        
    # Run the function
    master_data = create_master_dataset()
    
    # Optional: Print a summary of the unified behaviors
    print("\nFinal Behavior Distribution:")
    print(master_data['behavior_name'].value_counts())