# src/random_fields.py
import numpy as np
from scipy.fft import fft2, ifft2
import config
import matplotlib.pyplot as plt

class RandomFieldGenerator:
    def __init__(self):
        self.nx = config.MESH_SIZE
        self.nz = config.MESH_SIZE
        self.dx = config.DOMAIN_X / self.nx
        self.dz = config.DOMAIN_Z / self.nz

    def get_lognormal_params(self, mean, cov):
        """Converts normal mean/cov to lognormal parameters (Eq 2 & 3 in paper)."""
        std = mean * cov
        var = std**2
        mu_ln = np.log(mean**2 / np.sqrt(var + mean**2))
        sig_ln = np.sqrt(np.log(1.0 + var / mean**2))
        return mu_ln, sig_ln

    def generate_gaussian_field(self):
        """Generates a standard 2D Gaussian random field using FFT."""
        # Create physical coordinate grids
        x = np.arange(0, self.nx * self.dx, self.dx)
        z = np.arange(0, self.nz * self.dz, self.dz)
        X, Z = np.meshgrid(x, z)

        # Squared Exponential Autocorrelation Function (Eq 4 in paper)
        R = np.exp(-((X / config.L_H)**2 + (Z / config.L_V)**2))
        
        # Spectral decomposition via FFT
        S = np.abs(fft2(R))
        
        # Generate white noise
        noise = np.random.normal(0, 1, (self.nz, self.nx))
        noise_fft = fft2(noise)
        
        # Filter the noise with the spectral density and invert
        field_fft = noise_fft * np.sqrt(S)
        field = np.real(ifft2(field_fft))
        
        # Normalize to zero mean, unit variance
        return (field - np.mean(field)) / np.std(field)

    # NEW: Now requires the dynamic scalar inputs
    def generate_soil_sample(self, mean_c, mean_phi):
        """Generates the 4-channel tensor [c, phi, gamma, ks] for one slope."""
        
        # 1. Generate independent standard Gaussian fields
        G_c = self.generate_gaussian_field()
        G_phi_indep = self.generate_gaussian_field()
        G_gamma = self.generate_gaussian_field()
        G_ks = self.generate_gaussian_field()

        # 2. Induce cross-correlation between c and phi (Eq 5 in paper)
        # Pull the fixed CoVs from the new config
        cov_c, cov_phi = config.COV_C, config.COV_PHI
        rho_ln = np.log(1 + config.CORR_C_PHI * cov_c * cov_phi) / \
                 np.sqrt(np.log(1 + cov_c**2) * np.log(1 + cov_phi**2))
        
        # Correlate the Gaussian field for phi
        G_phi = rho_ln * G_c + np.sqrt(1 - rho_ln**2) * G_phi_indep

        # 3. Transform standard Gaussian to Lognormal physical values
        def to_lognormal(G, mean, cov):
            mu_ln, sig_ln = self.get_lognormal_params(mean, cov)
            return np.exp(mu_ln + sig_ln * G)

        # Use the dynamic means and fixed CoVs for C and Phi
        field_c = to_lognormal(G_c, mean_c, config.COV_C)
        field_phi = to_lognormal(G_phi, mean_phi, config.COV_PHI)
        
        # Gamma and Ks still use their static stats from config
        field_gamma = to_lognormal(G_gamma, config.STATS_GAMMA[0], config.STATS_GAMMA[1])
        field_ks = to_lognormal(G_ks, config.STATS_KS[0], config.STATS_KS[1])

        # Stack into a (4, 128, 128) tensor ready for the CNN/PINN
        return np.stack([field_c, field_phi, field_gamma, field_ks], axis=0)

if __name__ == "__main__":
    # Test the generator with example universal parameters
    generator = RandomFieldGenerator()
    sample = generator.generate_soil_sample(mean_c=25.0, mean_phi=30.0)
    print(f"Generated soil sample shape: {sample.shape}")
    print(f"Channels: 0: Cohesion, 1: Friction, 2: Unit Weight, 3: Permeability")