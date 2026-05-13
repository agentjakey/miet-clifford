"""
CLI driver for full (L, p) grid sweep.
Saves results to data/sweep_results.npz.
"""
import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.simulation import run_sweep

L_VALUES = [8, 12, 16, 24, 32]
P_VALUES = np.linspace(0.0, 0.5, 26)
N_LAYERS = 6
N_SAMPLES = 200


def main():
    parser = argparse.ArgumentParser(description="MIET full (L, p) sweep")
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    parser.add_argument("--n_layers", type=int, default=N_LAYERS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="../data/sweep_results.npz")
    args = parser.parse_args()

    out_path = os.path.join(os.path.dirname(__file__), args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    S_avg, S_std = run_sweep(
        L_VALUES, P_VALUES, args.n_layers, args.n_samples, seed=args.seed
    )
    np.savez(out_path, L=L_VALUES, p=P_VALUES, S_avg=S_avg, S_std=S_std)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
