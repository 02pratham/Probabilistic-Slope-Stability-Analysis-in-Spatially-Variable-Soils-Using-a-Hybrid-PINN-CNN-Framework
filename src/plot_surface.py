# src/plot_surface.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import RadioButtons
import torch
import sys, os

sys.path.append(os.path.abspath('./src'))
from src import config
from src.limit_analysis import trace_slip_surface, calculate_external_work, SlopeStabilitySolver, get_surface_elevation
from src.pinn_seepage import SeepageSolver

def visualize_multiple_surfaces(sample_indices=[0], train_dir="data/train"):
    print("Loading compiled master data into RAM...")
    try:
        X_data = np.load(os.path.join(train_dir, "X_train.npy"))
        y_data = np.load(os.path.join(train_dir, "y_train.npy"))
        S_data = np.load(os.path.join(train_dir, "S_train.npy")) 
    except FileNotFoundError:
        print(f"Error: Could not find master data in {train_dir}. Did you run --compile?")
        return

    for sample_idx in sample_indices:
        print(f"\n{'='*50}")
        print(f"Processing Sample {sample_idx}...")
        
        if sample_idx >= len(X_data):
            print(f"Error: Sample index {sample_idx} is out of bounds. Skipping.")
            continue

        sample = X_data[sample_idx]
        target_fos = y_data[sample_idx]
        current_beta = float(S_data[sample_idx][0])
        
        field_c = sample[0]
        field_phi = sample[1]
        field_gamma = sample[2]
        field_ks = sample[3]
        
        # Dictionary holding all property data, colormaps, and labels
        properties = {
            "Cohesion": {"data": field_c, "cmap": "YlOrRd", "label": "Soil Cohesion (kPa)"},
            "Friction": {"data": field_phi, "cmap": "YlGnBu", "label": "Friction Angle (degrees)"},
            "Gamma": {"data": field_gamma, "cmap": "Oranges", "label": "Unit Weight (kN/m^3)"},
            "Permeability": {"data": field_ks, "cmap": "Blues", "label": "Permeability (m/s)"}
        }

        print(f"Waking up PINN to calculate seepage forces (Slope Angle: {current_beta}°)...")
        solver_pinn = SeepageSolver(field_ks, beta=current_beta)
        solver_pinn.train(epochs=1000)

        print("Tracing the failure surfaces (This searches base, toe, and face failures)...")
        solver = SlopeStabilitySolver(sample, pinn_model=solver_pinn.model, beta=current_beta)
        
        theta_range = np.linspace(np.radians(30), np.radians(85), 20)
        r0_range = np.linspace(config.H * 0.8, config.H * 2.5, 20)
        
        crest_x = config.H / np.tan(np.radians(current_beta))
        x_start_range = np.linspace(-config.H * 0.5, crest_x * 0.8, 10) 
        
        evaluated_surfaces = []
        
        for x_start in x_start_range:
            z_start = get_surface_elevation(x_start, config.H, current_beta)
            
            for theta0 in theta_range:
                for r0 in r0_range:
                    if z_start + r0 * np.sin(theta0) <= config.H:
                        continue
                        
                    F_low, F_high = 0.5, 3.0
                    tol = 0.01
                    
                    last_valid_x = None
                    last_valid_z = None
                    
                    while (F_high - F_low) > tol:
                        F_mid = (F_low + F_high) / 2.0
                        
                        D, slip_x, slip_z = trace_slip_surface(
                            r0, theta0, F_mid, 
                            solver.field_c, solver.field_phi, 
                            solver.dx, solver.dz, solver.nx, solver.nz, 
                            solver.x_min, solver.z_min, current_beta,
                            x_start, z_start
                        )
                        
                        if D == -1.0: 
                            F_low = F_mid
                            continue 
                        
                        last_valid_x = slip_x
                        last_valid_z = slip_z
                        
                        W_gamma = calculate_external_work(
                            slip_x, slip_z, r0, theta0, 
                            solver.field_gamma, solver.dx, solver.dz, 
                            solver.nx, solver.nz, solver.x_min, solver.z_min, 
                            config.H, current_beta, x_start
                        )
                        
                        x_c = x_start - r0 * np.cos(theta0)
                        z_c = z_start + r0 * np.sin(theta0)
                        r_pts = np.sqrt((slip_x - x_c)**2 + (slip_z - z_c)**2)
                        
                        W_seepage = solver.get_seepage_work(slip_x, slip_z, r_pts)
                        Total_Work = W_gamma + W_seepage

                        if Total_Work > D:
                            F_high = F_mid
                        else:
                            F_low = F_mid
                            
                    if last_valid_x is not None and F_high < 2.9: 
                        evaluated_surfaces.append((F_high, last_valid_x, last_valid_z))

        evaluated_surfaces.sort(key=lambda x: x[0])
        
        if not evaluated_surfaces:
            print("No valid failure surfaces found for this sample!")
            continue
            
        min_global_FoS = evaluated_surfaces[0][0]
        print(f"Plotting! Target FoS from Dataset: {target_fos:.3f} | Re-calculated Critical FoS: {min_global_FoS:.3f}")

        # --- Plotting Setup ---
        fig, ax = plt.subplots(figsize=(14, 8))
        # Compress the main plot slightly to the right to make room for the control panel
        fig.subplots_adjust(left=0.22) 
        
        extent = [solver.x_min, solver.x_min + config.DOMAIN_X, 
                  solver.z_min, solver.z_min + config.DOMAIN_Z]
                  
        # Draw the initial background property (Cohesion)
        initial_prop = "Cohesion"
        im = ax.imshow(properties[initial_prop]["data"], cmap=properties[initial_prop]["cmap"], 
                       extent=extent, origin='lower', alpha=0.6) 
        cbar = fig.colorbar(im, ax=ax, label=properties[initial_prop]["label"])

        # Draw the ground surface
        ax.plot([solver.x_min, 0, crest_x, solver.x_min + config.DOMAIN_X], 
                [0, 0, config.H, config.H], 'k-', linewidth=2, label='Ground Surface', zorder=5)

        # White out the air domain above the slope
        x_air = np.linspace(solver.x_min, solver.x_min + config.DOMAIN_X, 500)
        z_surf = [get_surface_elevation(x, config.H, current_beta) for x in x_air]
        ax.fill_between(x_air, z_surf, solver.z_min + config.DOMAIN_Z, color='white', alpha=1.0, zorder=4)

        # Plot the failure surfaces
        num_to_plot = min(50, len(evaluated_surfaces))
        indices_to_plot = np.linspace(0, num_to_plot - 1, 5, dtype=int)
        colors = cm.viridis(np.linspace(0, 1, len(indices_to_plot)))

        for idx, color in zip(reversed(indices_to_plot), reversed(colors)):
            fos, sx, sz = evaluated_surfaces[idx]
            
            if idx == 0:
                ax.plot(sx, sz, color='blue', linestyle='--', linewidth=3.5, label=f'Critical Surface (FoS={fos:.2f})', zorder=6)
            else:
                ax.plot(sx, sz, color=color, linestyle='-', linewidth=1.5, alpha=0.8, label=f'Sub-Critical (FoS={fos:.2f})', zorder=5)

        # Draw the Info Box on the main plot axes
        info_text = (
            f"--- Slope Parameters ---\n"
            f"Sample ID: {sample_idx}\n"
            f"Slope Angle (β): {current_beta}°\n"
            f"Slope Height: {config.H} m\n"
            f"Target FoS: {target_fos:.3f}\n"
            f"Recalculated Min FoS: {min_global_FoS:.3f}"
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
        ax.text(0.02, 0.96, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, zorder=10)

        # --- Interactive Radio Buttons ---
        # Create an axes for the radio buttons in the blank space on the left
        rax = fig.add_axes([0.02, 0.45, 0.15, 0.15], facecolor='white')
        rax.set_title('Toggle Background', fontweight='bold')
        radio = RadioButtons(rax, ('Cohesion', 'Friction', 'Gamma', 'Permeability'))

        # Callback function to update the background when clicked
        def update_bg(label):
            prop = properties[label]
            im.set_data(prop["data"])
            im.set_cmap(prop["cmap"])
            # Essential: Recalibrate the color scale bounds for the new data (e.g. going from 30kPa to 1e-6 m/s)
            im.set_clim(vmin=np.min(prop["data"]), vmax=np.max(prop["data"]))
            cbar.set_label(prop["label"])
            cbar.update_normal(im) 
            fig.canvas.draw_idle()

        # Connect the click event to our function
        radio.on_clicked(update_bg)

        # Formatting
        ax.set_xlim([-config.H * 0.5, crest_x + (config.H * 1.5)])
        ax.set_ylim([-config.H * 0.5, config.H + 1.0])
        ax.set_title(f"Band of Failure Mechanisms (Sample {sample_idx})")
        ax.set_xlabel("Horizontal Distance (m)")
        ax.set_ylabel("Elevation (m)")
        ax.legend(loc="upper right", framealpha=0.9)
        
        plt.show()

if __name__ == "__main__":
    indices_to_plot = [0] 
    visualize_multiple_surfaces(sample_indices=indices_to_plot)