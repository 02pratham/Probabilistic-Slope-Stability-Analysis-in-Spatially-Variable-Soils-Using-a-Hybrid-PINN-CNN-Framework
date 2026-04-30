# src/pinn_seepage.py
import torch
import torch.nn as nn
import numpy as np
import math
from scipy.stats import qmc
from src import config

if torch.cuda.is_available():
    torch.cuda.init()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PINN(nn.Module):
    def __init__(self, layers=[2, 50, 50, 50, 50, 1]):
        super(PINN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(layers[0], layers[1]))
        
        for i in range(1, len(layers) - 2):
            self.hidden_layers.append(nn.Linear(layers[i], layers[i+1]))
            
        self.output_layer = nn.Linear(layers[-2], layers[-1])
        self.activation = nn.Tanh()

    def forward(self, x, z):
        inputs = torch.cat([x, z], dim=1)
        for layer in self.hidden_layers:
            inputs = self.activation(layer(inputs))
        return self.output_layer(inputs)

class SeepageSolver:
    def __init__(self, ks_grid, beta=25.0):
        self.model = PINN().to(device)
        
        # FIX: Defined both Adam and L-BFGS optimizers for two-phase training
        self.optimizer_adam = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.optimizer_lbfgs = torch.optim.LBFGS(
            self.model.parameters(), 
            max_iter=500, 
            tolerance_grad=1e-7, 
            history_size=50
        )
        
        self.ks_grid = torch.tensor(ks_grid, dtype=torch.float32).to(device)
        self.lambda_pde = 1e12  
        
        self.beta = beta # Store dynamic beta
        
        # Slope geometry boundaries updated to use dynamic beta
        self.x_min, self.x_max = -config.H, config.H / np.tan(np.radians(self.beta)) + 2*config.H
        self.z_min, self.z_max = -config.H, config.H
        self.toe_x = 0.0
        self.crest_x = config.H / np.tan(np.radians(self.beta))

    def get_collocation_points(self, n_pde=5000, n_bc=1000):
        sampler = qmc.Sobol(d=2, scramble=True)
        
        pde_pts = []
        
        # Calculate the exponent (m) for the next highest power of 2
        m = math.ceil(math.log2(n_pde))
        
        while len(pde_pts) < n_pde:
            # Draw 2^m samples to maintain perfect mathematical balance
            sample = sampler.random_base2(m=m)
            
            xs = self.x_min + sample[:, 0] * (self.x_max - self.x_min)
            zs = self.z_min + sample[:, 1] * (self.z_max - self.z_min)
            
            for x, z in zip(xs, zs):
                # Slope surface logic uses dynamic beta
                if x < self.toe_x and z > 0: continue
                if self.toe_x <= x <= self.crest_x and z > x * np.tan(np.radians(self.beta)): continue
                pde_pts.append([x, z])
                if len(pde_pts) == n_pde: break
                
        pde_pts = torch.tensor(pde_pts, dtype=torch.float32, requires_grad=True).to(device)
        x_pde, z_pde = pde_pts[:, 0:1], pde_pts[:, 1:2]
        
        bc_top_x = torch.linspace(self.x_min, self.x_max, n_bc//2).unsqueeze(1)
        # Uses dynamic beta
        bc_top_z = torch.where(bc_top_x < self.toe_x, torch.zeros_like(bc_top_x),
                   torch.where(bc_top_x < self.crest_x, bc_top_x * np.tan(np.radians(self.beta)), 
                   torch.full_like(bc_top_x, config.H)))
                   
        bc_top_x = bc_top_x.clone().detach().requires_grad_(True).to(device)
        bc_top_z = bc_top_z.clone().detach().requires_grad_(True).to(device)

        return (x_pde, z_pde), (bc_top_x, bc_top_z)

    def interpolate_ks(self, x, z):
        norm_x = (x - self.x_min) / (self.x_max - self.x_min) * (config.MESH_SIZE - 1)
        norm_z = (z - self.z_min) / (self.z_max - self.z_min) * (config.MESH_SIZE - 1)
        
        idx_x = torch.clamp(norm_x.long(), 0, config.MESH_SIZE - 1)
        idx_z = torch.clamp(norm_z.long(), 0, config.MESH_SIZE - 1)
        return self.ks_grid[idx_z, idx_x]

    def compute_loss(self, pde_pts, bc_top_pts):
        x_pde, z_pde = pde_pts
        bc_top_x, bc_top_z = bc_top_pts

        h = self.model(x_pde, z_pde)
        
        h_x = torch.autograd.grad(h, x_pde, grad_outputs=torch.ones_like(h), create_graph=True)[0]
        h_z = torch.autograd.grad(h, z_pde, grad_outputs=torch.ones_like(h), create_graph=True)[0]
        
        ks = self.interpolate_ks(x_pde, z_pde)
        
        ks_h_x = ks * h_x
        ks_h_z = ks * h_z
        
        pde_residual_x = torch.autograd.grad(ks_h_x, x_pde, grad_outputs=torch.ones_like(ks_h_x), create_graph=True)[0]
        pde_residual_z = torch.autograd.grad(ks_h_z, z_pde, grad_outputs=torch.ones_like(ks_h_z), create_graph=True)[0]
        
        loss_pde = torch.mean((pde_residual_x + pde_residual_z)**2)

        h_top = self.model(bc_top_x, bc_top_z)
        loss_bc_top = torch.mean((h_top - bc_top_z)**2)
        
        total_loss = self.lambda_pde * loss_pde + loss_bc_top
        return total_loss

    def train(self, epochs=1000):
        pde_pts, bc_top_pts = self.get_collocation_points()
        loss_history = [] 
        
        # Phase 1: Adam Optimizer (Fast initial convergence)
        for epoch in range(1, epochs+1):
            self.optimizer_adam.zero_grad()
            loss = self.compute_loss(pde_pts, bc_top_pts)
            loss.backward()
            self.optimizer_adam.step()
            
            loss_history.append(loss.item()) 
            
        # FIX: Phase 2: L-BFGS fine-tuning (Second-order precision)
        def closure():
            self.optimizer_lbfgs.zero_grad()
            loss = self.compute_loss(pde_pts, bc_top_pts)
            loss.backward()
            return loss
            
        self.optimizer_lbfgs.step(closure)
        
        # Append the final refined loss
        final_loss = self.compute_loss(pde_pts, bc_top_pts).item()
        loss_history.append(final_loss)
                
        return loss_history 

if __name__ == "__main__":
    from src.random_fields import RandomFieldGenerator
    gen = RandomFieldGenerator()
    sample = gen.generate_soil_sample(mean_c=25.0, mean_phi=30.0)
    ks_field = sample[3] 
    
    print("Training PINN Seepage Solver...")
    solver = SeepageSolver(ks_field, beta=35.0)
    solver.train(epochs=500)