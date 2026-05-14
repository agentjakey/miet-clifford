"""
Figure 0: Hybrid brickwork circuit architecture schematic (Methods figure).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os

mpl.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 300,
    }
)

OUT_PDF = os.path.join(
    os.path.dirname(__file__), "../figures/fig0_circuit_schematic.pdf"
)
OUT_PNG = os.path.join(
    os.path.dirname(__file__), "../figures/fig0_circuit_schematic.png"
)

N_QUBITS = 6
N_LAYERS = 4
GATE_W = 0.35
GATE_PAD = 0.05

MEAS = {
    0: [1, 4],
    1: [2],
    2: [0, 3],
    3: [5],
}


def pairs(t):
    start = t % 2
    return [(q, q + 1) for q in range(start, N_QUBITS - 1, 2)]


def main():
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.set_xlim(-0.7, N_LAYERS - 0.3)
    ax.set_ylim(-0.6, N_QUBITS - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("Circuit layer $t$")
    ax.set_ylabel("Qubit index")
    ax.set_xticks(range(N_LAYERS))
    ax.set_yticks(range(N_QUBITS))
    ax.set_yticklabels([f"$q_{q}$" for q in range(N_QUBITS)])
    ax.tick_params(axis="x", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    wire_end = N_LAYERS - 0.3
    for q in range(N_QUBITS):
        ax.plot([-0.65, wire_end], [q, q], color="black", lw=1.2, zorder=1)

    for t in range(N_LAYERS):
        for q0, q1 in pairs(t):
            x_lo = t - GATE_W
            y_lo = min(q0, q1) - GATE_PAD
            height = abs(q1 - q0) + 2 * GATE_PAD
            rect = mpatches.FancyBboxPatch(
                (x_lo, y_lo),
                2 * GATE_W,
                height,
                boxstyle="round,pad=0.03",
                facecolor="#AED6F1",
                edgecolor="black",
                linewidth=1.5,
                zorder=3,
            )
            ax.add_patch(rect)
            y_mid = (q0 + q1) / 2.0
            ax.text(
                t,
                y_mid,
                "C",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="navy",
                zorder=4,
            )

    meas_x_offset = GATE_W + 0.12
    for t, qubits in MEAS.items():
        x_m = t + meas_x_offset
        for q in qubits:
            ax.plot([x_m - 0.04, x_m + 0.04], [q, q], color="crimson", lw=2.0, zorder=5)
            ax.text(
                x_m,
                q - 0.22,
                "M",
                ha="center",
                va="bottom",
                fontsize=10,
                color="crimson",
                fontweight="bold",
                zorder=5,
            )
            ax.annotate(
                "",
                xy=(x_m, q + 0.22),
                xytext=(x_m, q - 0.0),
                arrowprops=dict(
                    arrowstyle="-|>", color="crimson", lw=1.2, mutation_scale=8
                ),
                zorder=5,
            )

    ax.set_title("Hybrid Brickwork Circuit with Measurements", pad=10)

    ax.text(
        0.18,
        0.97,
        r"Volume-law: $p < p_c \approx 0.16$",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        color="#1A5276",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#D6EAF8",
            edgecolor="#1A5276",
            alpha=0.85,
        ),
    )
    ax.text(
        0.82,
        0.97,
        r"Area-law: $p > p_c$",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="right",
        color="#7B241C",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#FADBD8",
            edgecolor="#7B241C",
            alpha=0.85,
        ),
    )

    legend_elements = [
        mpatches.Patch(
            facecolor="#AED6F1",
            edgecolor="black",
            linewidth=1.5,
            label="Random 2-qubit Clifford (C)",
        ),
        Line2D(
            [0],
            [0],
            marker="$M$",
            color="crimson",
            markersize=9,
            linestyle="None",
            label="Projective measurement (prob. $p$)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()


# ============================================================
# LaTeX caption -- paste into main.tex \caption{...}
# ============================================================
# Hybrid brickwork circuit architecture. Blue boxes labeled \textbf{C} represent
# random two-qubit Clifford gates; red \textbf{M} symbols are single-site
# projective measurements applied independently with probability $p$.
# Even layers pair qubits $(0,1),(2,3),(4,5)$; odd layers pair $(1,2),(3,4)$,
# with unpaired boundary qubits receiving a random single-qubit Clifford.
# The competition between unitary scrambling (C gates) and projective collapse
# (M measurements) drives the measurement-induced phase transition at $p = p_c$.
