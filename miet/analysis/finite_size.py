"""
Figure 3: Finite-size scaling collapse.
Rescales x -> (p - p_c) * L^(1/nu) and plots S/S_max vs rescaled variable.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

DATA = os.path.join(os.path.dirname(__file__), "../data/sweep_results.npz")
OUT = os.path.join(os.path.dirname(__file__), "../figures/finite_size")

P_C = 0.16
NU = 1.3


def main():
    d = np.load(DATA)
    L_values, p_values, S_avg = d["L"], d["p"], d["S_avg"]

    fig, ax = plt.subplots(figsize=(5, 4))
    for i, L in enumerate(L_values):
        x = (p_values - P_C) * (L ** (1.0 / NU))
        S = S_avg[i]
        S_max = S.max()
        ax.plot(x, S / S_max, marker="o", markersize=3, label=f"L={L}")

    ax.set_xlabel(r"$(p - p_c) L^{1/\nu}$")
    ax.set_ylabel(r"$S / S_{\max}$")
    ax.set_title(f"Finite-size scaling collapse (p_c={P_C}, nu={NU})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT + ".pdf")
    fig.savefig(OUT + ".png", dpi=150)
    print(f"Saved {OUT}.pdf and .png")


if __name__ == "__main__":
    main()
