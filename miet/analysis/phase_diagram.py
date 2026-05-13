"""
Figure 1: S vs p crossing plot (phase diagram).
Loads data/sweep_results.npz and plots S(p) for each L.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

DATA = os.path.join(os.path.dirname(__file__), "../data/sweep_results.npz")
OUT = os.path.join(os.path.dirname(__file__), "../figures/phase_diagram")


def main():
    d = np.load(DATA)
    L_values, p_values, S_avg = d["L"], d["p"], d["S_avg"]

    fig, ax = plt.subplots(figsize=(5, 4))
    for i, L in enumerate(L_values):
        ax.plot(p_values, S_avg[i], marker="o", markersize=3, label=f"L={L}")
    ax.set_xlabel("Measurement rate p")
    ax.set_ylabel("S(L/2)")
    ax.set_title("Entanglement entropy vs measurement rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT + ".pdf")
    fig.savefig(OUT + ".png", dpi=150)
    print(f"Saved {OUT}.pdf and .png")


if __name__ == "__main__":
    main()
