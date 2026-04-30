# src/generate_raw_data.py
import numpy as np
import os
import torch
from tqdm import tqdm
from src.random_fields import RandomFieldGenerator
from src import config

if torch.cuda.is_available():
    torch.cuda.init()

def generate_raw_fields(num_samples=400, output_dir="data/raw", seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    os.makedirs(output_dir, exist_ok=True)
    
    generator = RandomFieldGenerator()
    
    print(f"Generating {num_samples} Universal soil fields (Seed: {seed})...")
    for i in tqdm(range(num_samples)):
        # 1. Randomly sample from our universal lists
        beta = np.random.choice(config.SLOPE_ANGLES)
        mean_c = np.random.choice(config.COHESION_MEANS)
        mean_phi = np.random.choice(config.FRICTION_MEANS)
        
        # Package them into a 1D array
        scalars = np.array([beta, mean_c, mean_phi], dtype=np.float32)
        
        # 2. Generate the spatial tensor with dynamic inputs
        sample_tensor = generator.generate_soil_sample(mean_c=mean_c, mean_phi=mean_phi)
        
        # 3. Save BOTH the tensor and the scalars together as an .npz!
        filename = f"sample_{i:04d}.npz"
        filepath = os.path.join(output_dir, filename)
        np.savez(filepath, field=sample_tensor, scalars=scalars)
        
    print(f"Successfully saved {num_samples} Universal samples to {output_dir}/")

if __name__ == "__main__":
    generate_raw_fields(num_samples=400, seed=42)