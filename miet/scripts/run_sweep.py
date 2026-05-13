#!/usr/bin/env python3
import sys, time, numpy as np
sys.path.insert(0, '.')
from src.simulation import parameter_sweep, load_results

L_VALUES  = [8, 12, 16, 20, 24]
P_VALUES  = np.linspace(0.0, 0.50, 26)
N_SAMPLES = 200
N_WORKERS = 1
SAVE_PATH = "data/sweep_results.npz"

print("="*60)
print("MIET Production Sweep")
print(f"  L values:  {L_VALUES}")
print(f"  p range:   {len(P_VALUES)} points in [0.00, 0.50]")
print(f"  Samples:   {N_SAMPLES} per (L, p)")
print(f"  Total:     {len(L_VALUES)*len(P_VALUES)*N_SAMPLES:,} circuit runs")
print("="*60)

t0      = time.time()
results = parameter_sweep(L_VALUES, P_VALUES, N_SAMPLES,
                          save_path=SAVE_PATH, verbose=True)
elapsed = time.time() - t0
print(f"\nCompleted in {elapsed/60:.1f} minutes. Saved to {SAVE_PATH}")

L_vals, p_vals, mean_S, _, _ = load_results(SAVE_PATH)
dS      = np.abs(np.diff(mean_S[-1]))
p_c_est = float(p_vals[np.argmax(dS)])
print(f"Estimated p_c (max |dS/dp| for L={L_VALUES[-1]}): {p_c_est:.3f}")
