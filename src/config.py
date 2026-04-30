# src/config.py
import numpy as np

# --- 1. Universal Macro Geometry ---
H = 5.0                 # Slope height (m) - Kept constant for the surrogate boundary
MESH_SIZE = 128         # Grid size for the CNN input (128x128)
DOMAIN_X = 20.0         # Physical width of the modeled domain (m)
DOMAIN_Z = 10.0         # Physical height of the modeled domain (m)

# NEW: Universal Slope Angles (Ranging from gentle base-failures to steep face-failures)
SLOPE_ANGLES = [20.0, 25.0, 30.0, 35.0, 45.0, 55.0, 65.0]

# --- 2. Universal Soil Statistical Properties ---
# NEW: Lists of Means to randomly sample from for each generated field
COHESION_MEANS = [10.0, 20.0, 30.0, 45.0, 60.0]        # 5 = Sand behavior, 32 = Clay behavior
FRICTION_MEANS = [20.0, 25.0, 30.0, 35.0, 40.0] # Friction angles (degrees)

# Fixed Coefficients of Variation (CoV) to maintain consistent spatial variance
COV_C = 0.40            # Cohesion variability
COV_PHI = 0.20          # Friction variability
COV_GAMMA = 0.10        # Unit Weight variablility

# Kept static as these represent standard compacted embankment physics
# FIX: Linked directly to COV_GAMMA to prevent mathematical contradictions
STATS_GAMMA = (19.0, COV_GAMMA) # Unit weight (kN/m^3)
STATS_KS = (1.0e-6, 0.8)        # Saturated permeability (m/s)

# --- 3. Spatial Correlation Parameters ---
CORR_C_PHI = -0.5               # Cross-correlation (higher friction usually means lower cohesion)
L_H = 20.0                      # Horizontal autocorrelation length (m)
L_V = 2.0                       # Vertical autocorrelation length (m)