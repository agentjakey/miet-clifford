"""
Figure 0: Circuit architecture diagram.
Draws a schematic of the hybrid random Clifford + measurement circuit.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

OUT = os.path.join(os.path.dirname(__file__), "../figures/circuit_schematic")

N_QUBITS = 4
N_LAYERS = 3


def main():
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.set_xlim(-0.5, N_LAYERS + 0.5)
    ax.set_ylim(-0.5, N_QUBITS - 0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Qubit lines
    for q in range(N_QUBITS):
        ax.axhline(q, color="k", lw=1, zorder=0)

    # Gate and measurement boxes
    for t in range(N_LAYERS):
        # 2-qubit gates on alternating pairs
        for pair_start in range(t % 2, N_QUBITS - 1, 2):
            rect = mpatches.FancyBboxPatch(
                (t + 0.1, pair_start - 0.3), 0.3, 1.6,
                boxstyle="round,pad=0.05",
                facecolor="steelblue", edgecolor="k", lw=0.8, zorder=2
            )
            ax.add_patch(rect)
            ax.text(t + 0.25, pair_start + 0.5, "C", ha="center", va="center",
                    color="white", fontsize=8, zorder=3)
        # Measurements (meter symbol)
        for q in range(N_QUBITS):
            ax.plot(t + 0.6, q, "v", color="firebrick", markersize=6, zorder=3)

    ax.set_title("Hybrid random Clifford + measurement circuit")
    fig.tight_layout()
    fig.savefig(OUT + ".pdf")
    fig.savefig(OUT + ".png", dpi=150)
    print(f"Saved {OUT}.pdf and .png")


if __name__ == "__main__":
    main()
