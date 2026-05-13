"""
Extract p_c and critical exponent nu from finite-size crossing data.
Fits the crossing points S(L, p) = S(L', p) as a function of L.
"""
import numpy as np
from scipy.optimize import minimize
import os

DATA = os.path.join(os.path.dirname(__file__), "../data/sweep_results.npz")


def _scaling_cost(params, L_values, p_values, S_avg):
    """Cost: variance of the collapse after rescaling."""
    p_c, nu = params
    all_x, all_s = [], []
    for i, L in enumerate(L_values):
        x = (p_values - p_c) * (L ** (1.0 / nu))
        s = S_avg[i] / S_avg[i].max()
        all_x.extend(x.tolist())
        all_s.extend(s.tolist())
    all_x = np.array(all_x)
    all_s = np.array(all_s)
    # Sort by x and compute variance around a smooth fit
    order = np.argsort(all_x)
    s_sorted = all_s[order]
    # Penalize spread: use std of residuals from running mean
    window = max(3, len(s_sorted) // 20)
    residuals = []
    for k in range(len(s_sorted) - window):
        residuals.append(s_sorted[k] - s_sorted[k: k + window].mean())
    return np.std(residuals)


def main():
    d = np.load(DATA)
    L_values, p_values, S_avg = d["L"], d["p"], d["S_avg"]

    result = minimize(
        _scaling_cost,
        x0=[0.16, 1.3],
        args=(L_values, p_values, S_avg),
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 2000},
    )
    p_c, nu = result.x
    print(f"p_c = {p_c:.4f}")
    print(f"nu  = {nu:.4f}")
    print(f"optimizer converged: {result.success}")


if __name__ == "__main__":
    main()
