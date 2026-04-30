# src/compile_data.py
import os
import numpy as np
from tqdm import tqdm

def compile_training_data(processed_dir="data/processed", train_dir="data/train", test_dir="data/test", test_split=0.2, random_seed=42):
    # FIX: Normalize paths to absolute Windows paths to prevent [Errno 22]
    processed_dir = os.path.abspath(os.path.normpath(processed_dir))
    train_dir = os.path.abspath(os.path.normpath(train_dir))
    test_dir = os.path.abspath(os.path.normpath(test_dir))

    print("Compiling all universal checkpointed samples into master train/test arrays...")
    
    npz_files = sorted([f for f in os.listdir(processed_dir) if f.endswith('.npz')])
    
    if not npz_files:
        print(f"Error: No .npz files found in {processed_dir}.")
        return
        
    X_list = []
    S_list = []  # NEW: We must compile the scalars too!
    y_list = []
    
    for filename in tqdm(npz_files, desc="Loading Checkpoints"):
        filepath = os.path.join(processed_dir, filename)
        data = np.load(filepath)
        
        X_list.append(data['X'])
        S_list.append(data['S'])
        y_list.append(data['y'])
        
    X_all = np.array(X_list)
    S_all = np.array(S_list)
    y_all = np.array(y_list)
    
    total_samples = len(X_all)
    
    print(f"\nShuffling and splitting data (Test Size: {test_split*100}%)...")
    
    np.random.seed(random_seed)
    indices = np.random.permutation(total_samples)
    
    X_all = X_all[indices]
    S_all = S_all[indices]
    y_all = y_all[indices]
    
    split_idx = int(total_samples * (1.0 - test_split))
    
    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    S_train, S_test = S_all[:split_idx], S_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    np.save(os.path.join(train_dir, "X_train.npy"), X_train)
    np.save(os.path.join(train_dir, "S_train.npy"), S_train)
    np.save(os.path.join(train_dir, "y_train.npy"), y_train)
    
    np.save(os.path.join(test_dir, "X_test.npy"), X_test)
    np.save(os.path.join(test_dir, "S_test.npy"), S_test)
    np.save(os.path.join(test_dir, "y_test.npy"), y_test)
    
    print(f"\nCompilation Complete!")
    print(f"Total samples stitched together: {total_samples}")
    print(f"--------------------------------------------------")
    print(f"Training Features (X_train) shape: {X_train.shape}")
    print(f"Training Scalars  (S_train) shape: {S_train.shape}")
    print(f"Training Targets  (y_train) shape: {y_train.shape}")
    print(f"--------------------------------------------------")

if __name__ == "__main__":
    compile_training_data(test_split=0.2)