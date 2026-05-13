"""
Fast sanity check: small L only, few samples.
Prints S vs p to stdout so you can eyeball the transition.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.simulation import run_sweep

L_VALUES = [6, 8]
P_VALUES = np.linspace(0.0, 0.5, 11)
N_LAYERS = 4
N_SAMPLES = 20


def main():
    S_avg, S_std = run_sweep(L_VALUES, P_VALUES, N_LAYERS, N_SAMPLES, seed=42)
    print(f"{'p':>6}  " + "  ".join(f"S(L={L})" for L in L_VALUES))
    for j, p in enumerate(P_VALUES):
        row = "  ".join(f"{S_avg[i, j]:.3f}" for i in range(len(L_VALUES)))
        print(f"{p:6.3f}  {row}")


if __name__ == "__main__":
    main()
