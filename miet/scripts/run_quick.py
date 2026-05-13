#!/usr/bin/env python3
import sys, time, numpy as np
sys.path.insert(0, '.')
from src.simulation import parameter_sweep

L_VALUES  = [8, 12]
P_VALUES  = np.linspace(0.0, 0.50, 11)
N_SAMPLES = 30
SAVE_PATH = "data/quick_results.npz"

print("MIET Quick Sweep")
t0 = time.time()
parameter_sweep(L_VALUES, P_VALUES, N_SAMPLES, save_path=SAVE_PATH, verbose=True)
print(f"Done in {time.time()-t0:.1f}s. Saved to {SAVE_PATH}")
