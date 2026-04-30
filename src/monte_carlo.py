# src/monte_carlo.py
import torch
import numpy as np
import os
import time
import torch.multiprocessing as mp
from queue import Empty
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.cnn_surrogate import UniversalFoSSurrogate
from src.random_fields import RandomFieldGenerator
from src import config

if torch.cuda.is_available():
    torch.cuda.init()

# ---------------------------------------------------------
# PRODUCER: The CPU Workers
# ---------------------------------------------------------
def field_generator_worker(worker_id, batches_to_produce, batch_size, queue, mean_c, mean_phi):
    """Generates random spatial fields on CPU and pushes them to the Queue."""
    gen = RandomFieldGenerator()
    
    for _ in range(batches_to_produce):
        X_batch = np.zeros((batch_size, 4, config.MESH_SIZE, config.MESH_SIZE), dtype=np.float32)
        
        for i in range(batch_size):
            sample = gen.generate_soil_sample(mean_c=mean_c, mean_phi=mean_phi)
            X_batch[i] = sample
            
        queue.put(X_batch)
        
    # Poison Pill to signal completion
    queue.put(None)

# ---------------------------------------------------------
# CONSUMER: The Main GPU Process
# ---------------------------------------------------------
def run_monte_carlo(target_beta, target_c=25.0, target_phi=30.0, model_path=None, output_dir="data/simulated", total_samples=100000, batch_size=250):
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*50)
    print(f"   Monte Carlo Simulation | Beta = {target_beta}°")
    print("="*50)
    print(f"Device        : {device}")
    print(f"Total Samples : {total_samples:,}")
    print(f"Target Design : Beta = {target_beta}°, c = {target_c} kPa, phi = {target_phi}°")
    
    # 1. Load the Universal Model Architecture
    model = UniversalFoSSurrogate().to(device)
    
    # Auto-find the latest model if none is provided
    if model_path is None:
        model_dir = "data/models"
        model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pth')])
        if not model_files:
            raise FileNotFoundError("No trained CNN found! Run training first.")
        model_path = os.path.join(model_dir, model_files[-1])
        
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # Pre-allocate scalar batch tensor
    S_fixed = np.array([[target_beta, target_c, target_phi]], dtype=np.float32)
    S_batch_tensor = torch.tensor(np.repeat(S_fixed, batch_size, axis=0)).to(device)
        
    # 2. Setup Multiprocessing Queue
    queue = mp.Queue(maxsize=20) 
    num_workers = 8
    total_batches = total_samples // batch_size
    batches_per_worker = total_batches // num_workers
    
    print(f"Spawning {num_workers} CPU Producers...")
    workers = []
    for i in range(num_workers):
        worker_batches = total_batches - (batches_per_worker * i) if i == num_workers - 1 else batches_per_worker
        p = mp.Process(target=field_generator_worker, args=(i, worker_batches, batch_size, queue, target_c, target_phi))
        p.start()
        workers.append(p)

    # 3. Consumer Inference Loop
    failures = 0
    fos_results = []
    active_workers = num_workers
    start_time = time.time()
    
    pbar = tqdm(total=total_batches, desc=f"Evaluating Beta={target_beta}°")

    with torch.no_grad():
        while active_workers > 0 or not queue.empty():
            try:
                X_batch_np = queue.get(timeout=10)
                
                if X_batch_np is None:
                    active_workers -= 1
                    continue
                
                X_batch_tensor = torch.tensor(X_batch_np).to(device)
                
                # Predict
                predictions = model(X_batch_tensor, S_batch_tensor).cpu().numpy().flatten()
                
                # Clamp rogue negatives
                predictions = np.maximum(predictions, 0.01)
                
                fos_results.extend(predictions)
                failures += np.sum(predictions < 1.0)
                pbar.update(1)
                
            except Empty:
                continue
                
    pbar.close()
    
    for p in workers:
        p.join()

    # 4. Calculate Final Reliability Metrics
    probability_of_failure = (failures / total_samples) * 100
    mean_fos = np.mean(fos_results)
    execution_time = time.time() - start_time
    
    # Save Results dynamically based on slope angle
    np.save(os.path.join(output_dir, f"mc_fos_distribution_beta{int(target_beta)}.npy"), np.array(fos_results))
    
    print(f"   -> Result: PoF = {probability_of_failure:.2f}% | Mean FoS = {mean_fos:.3f} | Speed: {(total_samples/execution_time):,.0f} slopes/sec")
    
    return fos_results

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    # --- AUTO-FIND TARGET MODEL FOR THIS SWEEP ---
    model_dir = "data/models"
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pth')])
    if not model_files:
        raise FileNotFoundError("No trained CNN found! Run training first.")
    
    target_model_path = os.path.join(model_dir, model_files[-1])
    # Extract just the name of the file without the .pth extension
    model_filename = os.path.basename(target_model_path).replace('.pth', '')
    
    # --- PARAMETRIC SWEEP ---
    betas_to_test = [25.0, 35.0, 45.0]
    results_dict = {}
    
    for beta in betas_to_test:
        # Pass the exact model path so all sweeps use the identical file
        fos_array = run_monte_carlo(target_beta=beta, total_samples=100_000, batch_size=250, model_path=target_model_path)
        results_dict[beta] = fos_array
        
    # --- PLOTTING THE SWEEP (HEADLESS MODE) ---
    print("\nGenerating Parametric Sweep Plot (Headless Mode)...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    colors = {25.0: 'mediumseagreen', 35.0: 'royalblue', 45.0: 'crimson'}
    
    for beta, fos_data in results_dict.items():
        pof = (np.sum(np.array(fos_data) < 1.0) / len(fos_data)) * 100
        sns.kdeplot(fos_data, fill=True, label=f'Beta = {beta}° (PoF: {pof:.1f}%)', color=colors[beta], alpha=0.5, linewidth=2)
        
    plt.axvline(x=1.0, color='black', linestyle='--', linewidth=2.5, label='Failure Threshold (FoS=1.0)')
    
    plt.title('Universal Surrogate: Probabilistic Slope Stability Sweep', fontsize=16, fontweight='bold')
    plt.xlabel('Calculated Factor of Safety (FoS)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.xlim(0.5, 3.0) 
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    
    # Save the plot directly using the dynamic model filename
    save_dir = "data/simulated"
    os.makedirs(save_dir, exist_ok=True)
    
    plot_path = os.path.join(save_dir, f"parametric_sweep_{model_filename}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Clear the matplotlib figure from memory
    plt.close()
    
    print(f"Plot saved successfully to: {plot_path}")