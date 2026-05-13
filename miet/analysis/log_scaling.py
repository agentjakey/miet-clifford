"""
Figure 2: S vs ln(|A|) at the critical point.
Fits S = (c/3) ln(|A|) + const to extract central charge c.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.simulation import run_single

OUT = os.path.join(os.path.dirname(__file__), "../figures/log_scaling")

L = 32
P_C = 0.16  # approximate critical point; refine with critical_fit.py
N_LAYERS = 6
N_SAMPLES = 100
SEED = 7


def main():
    rng = np.random.default_rng(SEED)
    subsystem_sizes = list(range(1, L // 2 + 1))
    S_vals = np.zeros(len(subsystem_sizes))

    for _ in range(N_SAMPLES):
        from src.stabilizer import StabilizerState
        from src.circuit import random_clifford_circuit
        from src.entropy import entanglement_entropy
        state = StabilizerState(L)
        seed = int(rng.integers(1 << 31))
        local_rng = np.random.default_rng(seed)
        for _ in range(N_LAYERS):
            random_clifford_circuit(state, P_C, rng=local_rng)
        for k, size in enumerate(subsystem_sizes):
            S_vals[k] += entanglement_entropy(state, list(range(size)))
    S_vals /= N_SAMPLES

    ln_A = np.log(subsystem_sizes)
    slope, intercept, r, *_ = linregress(ln_A, S_vals)
    c = 3 * slope

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(ln_A, S_vals, s=20, label="data")
    ax.plot(ln_A, slope * ln_A + intercept, "r--",
            label=f"fit: c/3={slope:.3f}, c={c:.3f}")
    ax.set_xlabel("ln(|A|)")
    ax.set_ylabel("S(A)")
    ax.set_title(f"Log scaling at p_c ~ {P_C} (L={L})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT + ".pdf")
    fig.savefig(OUT + ".png", dpi=150)
    print(f"Saved {OUT}.pdf and .png  [c = {c:.3f}]")


if __name__ == "__main__":
    main()
