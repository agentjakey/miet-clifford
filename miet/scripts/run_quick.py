#!/usr/bin/env python3
import sys, time, argparse, numpy as np
sys.path.insert(0, '.')
from src.simulation import parameter_sweep

L_VALUES  = [8, 12]
P_VALUES  = np.linspace(0.0, 0.50, 11)
N_SAMPLES = 30
SAVE_PATH = "data/quick_results.npz"


def main():
    parser = argparse.ArgumentParser(description="MIET quick sweep (small sizes)")
    parser.add_argument('--seed', type=int, default=None,
                        help='Master random seed for reproducibility (default: non-deterministic)')
    args = parser.parse_args()

    print(f"MIET Quick Sweep  (seed={args.seed if args.seed is not None else 'non-deterministic'})")
    t0 = time.time()
    parameter_sweep(L_VALUES, P_VALUES, N_SAMPLES,
                    save_path=SAVE_PATH, verbose=True, seed=args.seed)
    print(f"Done in {time.time()-t0:.1f}s. Saved to {SAVE_PATH}")


if __name__ == '__main__':
    main()
