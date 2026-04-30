# src/limit_analysis.py
import numpy as np
from numba import njit
import torch
from src import config

if torch.cuda.is_available():
    torch.cuda.init()

# Unit weight of water (kN/m^3)
GAMMA_W = 9.81

@njit
def bilinear_interpolate(grid, x, z, dx, dz, nx, nz, x_min, z_min):
    """Fast bilinear interpolation for spatial soil grids."""
    norm_x = (x - x_min) / dx
    norm_z = (z - z_min) / dz
    
    if norm_x < 0: norm_x = 0.0
    if norm_x > nx - 1.001: norm_x = nx - 1.001
    if norm_z < 0: norm_z = 0.0
    if norm_z > nz - 1.001: norm_z = nz - 1.001
    
    x1, z1 = int(norm_x), int(norm_z)
    x2, z2 = x1 + 1, z1 + 1
    
    wx2, wz2 = norm_x - x1, norm_z - z1
    wx1, wz1 = 1.0 - wx2, 1.0 - wz2
    
    val = (grid[z1, x1] * wx1 * wz1 +
           grid[z1, x2] * wx2 * wz1 +
           grid[z2, x1] * wx1 * wz2 +
           grid[z2, x2] * wx2 * wz2)
    return val

@njit
def get_surface_elevation(x, H, beta):
    """Returns the z-coordinate of the slope surface for any given x."""
    toe_x = 0.0
    crest_x = H / np.tan(np.radians(beta))
    
    if x < toe_x:
        return 0.0
    elif x < crest_x:
        return x * np.tan(np.radians(beta))
    else:
        return H

@njit
def calculate_external_work(slip_x, slip_z, r0, theta0, field_gamma, dx, dz, nx, nz, x_min, z_min, H, beta, x_start):
    """Integrates the work done by the soil's self-weight using vertical slices."""
    W_gamma = 0.0
    
    # Rotation center is relative to the dynamic starting point
    x_c = x_start - r0 * np.cos(theta0) 
    
    for i in range(len(slip_x) - 1):
        x1, z1 = slip_x[i], slip_z[i]
        x2, z2 = slip_x[i+1], slip_z[i+1]
        
        x_mid = (x1 + x2) / 2.0
        z_mid = (z1 + z2) / 2.0
        
        z_surf = get_surface_elevation(x_mid, H, beta)
        
        if z_mid >= z_surf:
            continue
            
        slice_width = x2 - x1
        slice_height = z_surf - z_mid
        slice_area = slice_width * slice_height
        
        gamma_local = bilinear_interpolate(field_gamma, x_mid, (z_mid + z_surf)/2.0, dx, dz, nx, nz, x_min, z_min)
        weight = slice_area * gamma_local
        
        v_y = x_mid - x_c
        W_gamma += weight * v_y
        
    return W_gamma

@njit
def trace_slip_surface(r0, theta0, F, field_c, field_phi, dx, dz, nx, nz, x_min, z_min, beta, x_start, z_start):
    """Traces the log-spiral failure mechanism."""
    delta = np.radians(0.1) 
    
    theta_i = theta0
    r_i = r0
    
    # Start the curve at the dynamic coordinates
    x_i = x_start
    z_i = z_start
    
    # Rotation center is relative to the dynamic starting point
    x_c = x_start - r0 * np.cos(theta0)
    z_c = z_start + r0 * np.sin(theta0)
    
    total_dissipation = 0.0
    
    max_pts = 2000
    slip_x = np.zeros(max_pts)
    slip_z = np.zeros(max_pts)
    slip_x[0] = x_i
    slip_z[0] = z_i
    pt_idx = 1
    
    while z_i < config.H:
        # Stop tracing immediately if the curve breaks out of the ground!
        z_surf = get_surface_elevation(x_i, config.H, beta)
        if z_i > z_surf + 0.05: 
            break
            
        c_local = bilinear_interpolate(field_c, x_i, z_i, dx, dz, nx, nz, x_min, z_min)
        phi_local = bilinear_interpolate(field_phi, x_i, z_i, dx, dz, nx, nz, x_min, z_min)
        
        c_f = c_local / F
        phi_f = np.arctan(np.tan(np.radians(phi_local)) / F)
        
        theta_next = theta_i - delta
        r_next = r_i * np.exp(delta * np.tan(phi_f)) 
        
        x_next = x_c + r_next * np.cos(theta_next)
        z_next = z_c - r_next * np.sin(theta_next)
        
        segment_length = np.sqrt((x_next - x_i)**2 + (z_next - z_i)**2)
        v_local = r_i 
        
        total_dissipation += c_f * v_local * np.cos(phi_f) * segment_length
        
        theta_i = theta_next
        r_i = r_next
        x_i = x_next
        z_i = z_next
        
        if pt_idx < max_pts:
            slip_x[pt_idx] = x_i
            slip_z[pt_idx] = z_i
            pt_idx += 1
            
        crest_x = config.H / np.tan(np.radians(beta))
        max_breakout_x = crest_x + (1.5 * config.H) 
        
        if x_i > max_breakout_x or z_i < z_min:
            return -1.0, slip_x[:1], slip_z[:1] 
            
    return total_dissipation, slip_x[:pt_idx], slip_z[:pt_idx]

class SlopeStabilitySolver:
    def __init__(self, sample, pinn_model=None, beta=25.0):
        self.field_c = sample[0]
        self.field_phi = sample[1]
        self.field_gamma = sample[2]
        self.pinn_model = pinn_model
        
        self.beta = beta
        
        self.nx = config.MESH_SIZE
        self.nz = config.MESH_SIZE
        self.dx = config.DOMAIN_X / self.nx
        self.dz = config.DOMAIN_Z / self.nz
        
        self.x_min = -config.H
        self.z_min = -config.H

    def get_seepage_work(self, slip_x, slip_z, r_pts):
        if self.pinn_model is None:
            return 0.0
            
        device = next(self.pinn_model.parameters()).device
            
        x_tensor = torch.tensor(slip_x, dtype=torch.float32).unsqueeze(1).to(device)
        z_tensor = torch.tensor(slip_z, dtype=torch.float32).unsqueeze(1).to(device)
        
        with torch.no_grad():
            h_pred = self.pinn_model(x_tensor, z_tensor).cpu().numpy().flatten()
            
        u = GAMMA_W * np.maximum(0.0, h_pred - slip_z)
        w_s = np.sum(u * r_pts * np.radians(0.1)) 
        return w_s

    def evaluate_FoS(self):
        theta_range = np.linspace(np.radians(30), np.radians(85), 20)
        r0_range = np.linspace(config.H * 0.8, config.H * 2.5, 20)
        
        crest_x = config.H / np.tan(np.radians(self.beta))
        
        # Standardized search range to match visualizer (80% up the slope)
        x_start_range = np.linspace(-config.H * 0.5, crest_x * 0.8, 10) 
        
        min_global_FoS = 10.0 
        
        for x_start in x_start_range:
            z_start = get_surface_elevation(x_start, config.H, self.beta)
            
            for theta0 in theta_range:
                for r0 in r0_range:
                    # Center of rotation must be above ground
                    if z_start + r0 * np.sin(theta0) <= config.H:
                        continue
                        
                    # FIX: Widened bounds to capture true extremes and unfurl the histogram spikes
                    F_low, F_high = 0.1, 5.0
                    tol = 0.01
                    
                    while (F_high - F_low) > tol:
                        F_mid = (F_low + F_high) / 2.0
                        
                        D, slip_x, slip_z = trace_slip_surface(
                            r0, theta0, F_mid, 
                            self.field_c, self.field_phi, 
                            self.dx, self.dz, self.nx, self.nz, 
                            self.x_min, self.z_min, self.beta,
                            x_start, z_start
                        )
                        
                        if D == -1.0: 
                            F_low = F_mid
                            continue 
                            
                        # FIX: True Depth Filter. Checks the absolute vertical span of the arc.
                        max_depth = np.max(slip_z) - np.min(slip_z)
                        if max_depth < 1.0:
                            F_low = F_mid
                            continue
                        
                        W_gamma = calculate_external_work(
                            slip_x, slip_z, r0, theta0, 
                            self.field_gamma, self.dx, self.dz, 
                            self.nx, self.nz, self.x_min, self.z_min, 
                            config.H, self.beta, x_start
                        )
                        
                        # Update seepage rotation center
                        x_c = x_start - r0 * np.cos(theta0)
                        z_c = z_start + r0 * np.sin(theta0)
                        
                        r_pts = np.sqrt((slip_x - x_c)**2 + (slip_z - z_c)**2)
                        W_seepage = self.get_seepage_work(slip_x, slip_z, r_pts)
                        
                        Total_Work = W_gamma + W_seepage

                        if Total_Work > D:
                            F_high = F_mid  
                        else:
                            F_low = F_mid   
                            
                    if F_high < min_global_FoS and D != -1.0:
                        min_global_FoS = F_high
                        
        return min_global_FoS

if __name__ == "__main__":
    from src.random_fields import RandomFieldGenerator
    import time
    
    print("Generating random field...")
    gen = RandomFieldGenerator()
    test_beta = 35.0
    sample = gen.generate_soil_sample(mean_c=25.0, mean_phi=30.0)
    
    print(f"Executing Universal Kinematic Limit Analysis for a {test_beta}-degree slope...")
    solver = SlopeStabilitySolver(sample, beta=test_beta)
    
    start_time = time.time()
    fos = solver.evaluate_FoS()
    end_time = time.time()
    
    print(f"Calculated Factor of Safety: {fos:.3f}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")