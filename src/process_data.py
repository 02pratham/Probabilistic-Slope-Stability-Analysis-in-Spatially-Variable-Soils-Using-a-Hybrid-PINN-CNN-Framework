# src/process_data.py
import numpy as np
import os
import torch
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from src.pinn_seepage import SeepageSolver
from src.limit_analysis import SlopeStabilitySolver

# Windows requires multiprocessing functions to be at the top level of the script
def process_single_sample(args):
    filepath, output_dir, filename = args
    try:
        # Re-initialize CUDA for this specific subprocess to avoid deadlocks
        if torch.cuda.is_available():
            torch.cuda.init()
            
        # 1. Load the bundled data
        data = np.load(filepath)
        sample_tensor = data['field']
        scalars = data['scalars']
        
        # Extract the specific beta for this slope (Index 0)
        current_beta = float(scalars[0])
        
        # 2. Train PINN on the GPU
        ks_field = sample_tensor[3]
        # Pass the dynamic beta to the PINN solver to correctly constrain the geometry!
        solver_pinn = SeepageSolver(ks_field, beta=current_beta)
        solver_pinn.train(epochs=1000) 
        pinn_model = solver_pinn.model
        
        # 3. Run Limit Analysis with the dynamic beta
        solver_stability = SlopeStabilitySolver(sample_tensor, pinn_model=pinn_model, beta=current_beta)
        fos = solver_stability.evaluate_FoS()
        
        # 4. Save the checkpoint (Tensor + Scalars + Target FoS)
        save_name = filename.replace('.npz', '_processed.npz')
        processed_filepath = os.path.join(output_dir, save_name)
        np.savez(processed_filepath, X=sample_tensor, S=scalars, y=fos)
        
        return True, filename
    except Exception as e:
        return False, f"Error in {filename}: {str(e)}"

def process_raw_data(raw_dir="data/raw", output_dir="data/processed", sample_range=None):
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Hardware Status ---")
    print(f"PINN Seepage Solver Device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Limit Analysis: CPU (Numba JIT Compiled)")
    print(f"-----------------------\n")
    
    # We are looking for .npz files now!
    raw_files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.npz')])
    if not raw_files:
        raise FileNotFoundError(f"No raw .npz files found in {raw_dir}. Run generate_raw.py first.")
        
    if sample_range is not None:
        start_idx, end_idx = sample_range
        raw_files = raw_files[start_idx:end_idx]
        print(f"Batch Mode Active: Processing samples from index {start_idx} to {end_idx-1}.")
        
    # Prepare the arguments for the multiprocessing pool
    tasks = [(os.path.join(raw_dir, f), output_dir, f) for f in raw_files]
    
    # CRITICAL: Cap the workers to avoid crashing the GPU. 
    # For a 6GB GPU, 4 is usually the sweet spot. If it crashes, lower it to 2 or 3.
    MAX_WORKERS = 8 
    
    print(f"Processing {len(raw_files)} Universal samples using {MAX_WORKERS} parallel workers...")
    
    success_count = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks and wrap with tqdm for a live progress bar
        futures = {executor.submit(process_single_sample, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Calculating Physics"):
            success, msg = future.result()
            if success:
                success_count += 1
            else:
                print(f"\n{msg}")
                
    print(f"\nProcessing Complete! Successfully processed {success_count}/{len(raw_files)} samples.")

if __name__ == "__main__":
    # Required for PyTorch to share CUDA tensors across Windows processes without deadlocking
    mp.set_start_method('spawn', force=True) 
    process_raw_data()