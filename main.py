# # main.py
# import argparse
# import time
# import os
# import sys

# import warnings
# warnings.filterwarnings("ignore")

# # Ensure Python can find our internal modules
# sys.path.append(os.path.abspath('./src'))

# from src.generate_raw_data import generate_raw_fields
# from src.process_data import process_raw_data
# from src.compile_data import compile_training_data
# from src.cnn_surrogate import train_surrogate_model
# from src.monte_carlo import run_monte_carlo
# from src.plot_surface import visualize_multiple_surfaces

# def print_banner():
#     print("="*60)
#     print("  Deep Learning for Probabilistic Slope Stability Analysis  ")
#     print("="*60)

# def main():
#     parser = argparse.ArgumentParser(description="Pipeline Orchestrator for Slope Stability DL Framework")
    
#     # Execution Flags
#     parser.add_argument('--all', action='store_true', help="Run the entire pipeline end-to-end")
#     parser.add_argument('--raw', action='store_true', help="Step 1: Generate raw random soil fields")
#     parser.add_argument('--process', action='store_true', help="Step 2: Run PINN and Limit Analysis physics solvers")
#     parser.add_argument('--compile', action='store_true', help="Step 3: Compile checkpoints into master dataset")
#     parser.add_argument('--train', action='store_true', help="Step 4: Train the CNN Surrogate model")
#     parser.add_argument('--mc', action='store_true', help="Step 5: Run Monte Carlo Simulation")
    
#     # Utility Flags
#     parser.add_argument('--plot', action='store_true', help="Utility: Visually plot failure mechanisms for specific samples")
    
#     # Hyperparameters & Configs
#     parser.add_argument('--num_samples', type=int, default=400, help="Number of training samples (default: 400)")
#     parser.add_argument('--mc_samples', type=int, default=500000, help="Number of Monte Carlo samples (default: 500k)")
#     parser.add_argument('--epochs', type=int, default=5000, help="Number of epochs for CNN training (default: 5000)")
    
#     # Advanced Targets
#     parser.add_argument('--process_range', type=int, nargs=2, metavar=('START', 'END'), 
#                         help="Process a specific range of samples (e.g., --process_range 0 100)")
#     parser.add_argument('--model_path', type=str, 
#                         help="Path to specific timestamped model for standalone Monte Carlo run")
#     parser.add_argument('--plot_indices', type=int, nargs='+', default=[0], 
#                         help="List of sample indices to plot (e.g., --plot_indices 0 10 250)")
    
#     args = parser.parse_args()
    
#     if not (args.all or args.raw or args.process or args.compile or args.train or args.mc or args.plot):
#         parser.print_help()
#         return

#     print_banner()
#     total_start_time = time.time()

#     raw_dir = "data/raw"
#     processed_dir = "data/processed"
#     train_dir = "data/train"
#     test_dir = "data/test"
#     simulated_dir = "data/simulated"
    
#     active_model_path = args.model_path 

#     # --- Step 1: Raw Data Generation ---
#     if args.all or args.raw:
#         print("\n[Step 1/5] Generating Raw Spatial Fields...")
#         start = time.time()
#         generate_raw_fields(num_samples=args.num_samples, output_dir=raw_dir)
#         print(f"Step 1 completed in {time.time() - start:.2f} seconds.")

#     # --- Step 2: Physics Processing (PINN + Limit Analysis) ---
#     if args.all or args.process:
#         print("\n[Step 2/5] Processing Data via Physics Solvers (GPU/CPU)...")
#         start = time.time()
#         process_raw_data(raw_dir=raw_dir, output_dir=processed_dir, sample_range=args.process_range)
#         print(f"Step 2 completed in {time.time() - start:.2f} seconds.")

#     # --- Step 3: Dataset Compilation ---
#     if args.all or args.compile:
#         print("\n[Step 3/5] Compiling Checkpoints into Master Dataset...")
#         start = time.time()
#         compile_training_data(processed_dir=processed_dir, train_dir=train_dir, test_dir=test_dir)
#         print(f"Step 3 completed in {time.time() - start:.2f} seconds.")

#     # --- Step 4: CNN Surrogate Training ---
#     if args.all or args.train:
#         print("\n[Step 4/5] Training Universal CNN Surrogate Model...")
#         start = time.time()
        
#         # NEW: Safety check for BOTH the image and scalar datasets
#         if not os.path.exists(os.path.join(train_dir, "X_train.npy")) or not os.path.exists(os.path.join(train_dir, "S_train.npy")):
#             print(f"Error: Universal master data (X_train or S_train) not found in {train_dir}!")
#             print("Run with --compile first to generate the new two-headed dataset.")
#             return
        
#         active_model_path = train_surrogate_model(train_dir=train_dir, test_dir=test_dir, epochs=args.epochs)
#         print(f"Step 4 completed in {time.time() - start:.2f} seconds.")

#     # --- Step 5: Monte Carlo Simulation ---
#     if args.all or args.mc:
#         print("\n[Step 5/5] Executing Monte Carlo Reliability Analysis...")
#         start = time.time()
        
#         if not active_model_path or not os.path.exists(active_model_path):
#             print("\nError: No valid model path provided for Monte Carlo!")
#             print("If running '--mc' alone, you MUST provide the specific path using the flag:")
#             print("Example: python main.py --mc --model_path data/models/cnn_model_20260427_173834.pth")
#             return
            
#         run_monte_carlo(model_path=active_model_path, output_dir=simulated_dir, total_samples=args.mc_samples)
#         print(f"Step 5 completed in {time.time() - start:.2f} seconds.")

#     # --- UTILITY: Plot Failure Surfaces ---
#     if args.plot:
#         print("\n[Utility] Visualizing Physical Failure Mechanisms...")
#         start = time.time()
#         visualize_multiple_surfaces(sample_indices=args.plot_indices, train_dir=train_dir)
#         print(f"Plotting session ended after {time.time() - start:.2f} seconds.")

#     if args.all or args.raw or args.process or args.compile or args.train or args.mc:
#         print("\n" + "="*60)
#         print(f"Pipeline Execution Complete! Total Time: {(time.time() - total_start_time)/60:.2f} minutes.")
#         print("="*60)

# if __name__ == "__main__":
#     main()


# main.py
import argparse
import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# Ensure Python can find our internal modules
sys.path.append(os.path.abspath('./src'))

from src.generate_raw_data import generate_raw_fields
from src.process_data import process_raw_data
from src.compile_data import compile_training_data
from src.cnn_surrogate import train_surrogate_model
from src.monte_carlo import run_monte_carlo
from src.plot_surface import visualize_multiple_surfaces

def print_banner():
    print("="*60)
    print("  Deep Learning for Probabilistic Slope Stability Analysis  ")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Pipeline Orchestrator for Slope Stability DL Framework")
    
    # Execution Flags
    parser.add_argument('--all', action='store_true', help="Run the entire pipeline end-to-end")
    parser.add_argument('--raw', action='store_true', help="Step 1: Generate raw random soil fields")
    parser.add_argument('--process', action='store_true', help="Step 2: Run PINN and Limit Analysis physics solvers")
    parser.add_argument('--compile', action='store_true', help="Step 3: Compile checkpoints into master dataset")
    parser.add_argument('--train', action='store_true', help="Step 4: Train the CNN Surrogate model")
    parser.add_argument('--mc', action='store_true', help="Step 5: Run Monte Carlo Simulation")
    
    # Utility Flags
    parser.add_argument('--plot', action='store_true', help="Utility: Visually plot failure mechanisms for specific samples")
    
    # Hyperparameters & Configs
    parser.add_argument('--num_samples', type=int, default=400, help="Number of training samples (default: 400)")
    parser.add_argument('--mc_samples', type=int, default=100000, help="Number of Monte Carlo samples (default: 100k)")
    parser.add_argument('--epochs', type=int, default=5000, help="Number of epochs for CNN training (default: 5000)")
    
    # Advanced Targets
    parser.add_argument('--process_range', type=int, nargs=2, metavar=('START', 'END'), 
                        help="Process a specific range of samples (e.g., --process_range 0 100)")
    parser.add_argument('--model_path', type=str, 
                        help="Path to specific timestamped model for standalone Monte Carlo run")
    parser.add_argument('--plot_indices', type=int, nargs='+', default=[0], 
                        help="List of sample indices to plot (e.g., --plot_indices 0 10 250)")
    
    args = parser.parse_args()
    
    if not (args.all or args.raw or args.process or args.compile or args.train or args.mc or args.plot):
        parser.print_help()
        return

    print_banner()
    total_start_time = time.time()

    raw_dir = "data/raw"
    processed_dir = "data/processed"
    train_dir = "data/train"
    test_dir = "data/test"
    simulated_dir = "data/simulated"
    
    active_model_path = args.model_path 

    # --- Step 1: Raw Data Generation ---
    if args.all or args.raw:
        print("\n[Step 1/5] Generating Raw Spatial Fields...")
        start = time.time()
        generate_raw_fields(num_samples=args.num_samples, output_dir=raw_dir)
        print(f"Step 1 completed in {time.time() - start:.2f} seconds.")

    # --- Step 2: Physics Processing (PINN + Limit Analysis) ---
    if args.all or args.process:
        print("\n[Step 2/5] Processing Data via Physics Solvers (GPU/CPU)...")
        start = time.time()
        process_raw_data(raw_dir=raw_dir, output_dir=processed_dir, sample_range=args.process_range)
        print(f"Step 2 completed in {time.time() - start:.2f} seconds.")

    # --- Step 3: Dataset Compilation ---
    if args.all or args.compile:
        print("\n[Step 3/5] Compiling Checkpoints into Master Dataset...")
        start = time.time()
        compile_training_data(processed_dir=processed_dir, train_dir=train_dir, test_dir=test_dir)
        print(f"Step 3 completed in {time.time() - start:.2f} seconds.")

    # --- Step 4: CNN Surrogate Training ---
    if args.all or args.train:
        print("\n[Step 4/5] Training Universal CNN Surrogate Model...")
        start = time.time()
        
        # Safety check for BOTH the image and scalar datasets
        if not os.path.exists(os.path.join(train_dir, "X_train.npy")) or not os.path.exists(os.path.join(train_dir, "S_train.npy")):
            print(f"Error: Universal master data (X_train or S_train) not found in {train_dir}!")
            print("Run with --compile first to generate the new two-headed dataset.")
            return
        
        active_model_path = train_surrogate_model(train_dir=train_dir, test_dir=test_dir, epochs=args.epochs)
        print(f"Step 4 completed in {time.time() - start:.2f} seconds.")

    # --- Step 5: Monte Carlo Parametric Sweep ---
    if args.all or args.mc:
        print("\n[Step 5/5] Executing Monte Carlo Reliability Analysis (Parametric Sweep)...")
        start = time.time()
        
        # Auto-find the latest model if one wasn't explicitly provided or just trained
        if not active_model_path:
            model_dir = "data/models"
            model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pth')])
            if not model_files:
                print("Error: No trained CNN found! Run '--train' first.")
                return
            active_model_path = os.path.join(model_dir, model_files[-1])
            
        model_filename = os.path.basename(active_model_path).replace('.pth', '')
        
        # Execute the Sweep across different slope angles
        betas_to_test = [25.0, 35.0, 45.0]
        results_dict = {}
        
        for beta in betas_to_test:
            fos_array = run_monte_carlo(
                target_beta=beta, 
                model_path=active_model_path, 
                output_dir=simulated_dir, 
                total_samples=args.mc_samples
            )
            results_dict[beta] = fos_array
            
        # Plotting the Sweep (Headless Mode)
        print("\nGenerating Parametric Sweep Plot...")
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
        
        os.makedirs(simulated_dir, exist_ok=True)
        plot_path = os.path.join(simulated_dir, f"parametric_sweep_{model_filename}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved successfully to: {plot_path}")
        print(f"Step 5 completed in {time.time() - start:.2f} seconds.")

    # --- UTILITY: Plot Failure Surfaces ---
    if args.plot:
        print("\n[Utility] Visualizing Physical Failure Mechanisms...")
        start = time.time()
        visualize_multiple_surfaces(sample_indices=args.plot_indices, train_dir=train_dir)
        print(f"Plotting session ended after {time.time() - start:.2f} seconds.")

    if args.all or args.raw or args.process or args.compile or args.train or args.mc:
        print("\n" + "="*60)
        print(f"Pipeline Execution Complete! Total Time: {(time.time() - total_start_time)/60:.2f} minutes.")
        print("="*60)

if __name__ == "__main__":
    main()