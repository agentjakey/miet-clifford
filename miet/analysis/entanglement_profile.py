"""
Figure 4: Entanglement profile S(A) vs subsystem size |A|.

For representative measurement rates below, near, and above the MIPT critical
point, plots the disorder-averaged entanglement entropy S(A) as a function of
subsystem size |A| = 1, 2, ..., L/2 for a fixed system size L.

Physics signatures:
  Volume-law phase (p << p_c): S(|A|) ~ s * |A|, linear growth.
  Area-law phase   (p >> p_c): S(|A|) ~ O(1), flat / weakly growing.
  Near-critical    (p ~ p_c):  S(|A|) ~ alpha * ln|A| + const, logarithmic.

This figure provides direct, subsystem-size-resolved evidence for the two
phases beyond the half-chain entropy alone.

Reference:
  Li, Chen, Fisher, PRB 100, 134306 (2019) -- Section III, entanglement profiles
  Hamma, Ionicioiu, Zanardi, PRA 71, 022315 (2005) -- entropy from GF(2) rank
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

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

from miet_clifford.circuit import run_circuit
from miet_clifford.entropy import entanglement_entropy

DATA_OUT = os.path.join(os.path.dirname(__file__), "../data/entanglement_profile.npz")
OUT_PDF = os.path.join(
    os.path.dirname(__file__), "../figures/fig4_entanglement_profile.pdf"
)
OUT_PNG = os.path.join(
    os.path.dirname(__file__), "../figures/fig4_entanglement_profile.png"
)

# Simulation parameters
L = 20
N_SAMPLES = 100
N_STEPS = 4 * L
WARMUP = 3 * L

# Three representative measurement rates
P_VALUES = [0.04, 0.16, 0.32]
P_LABELS = [
    r"$p = 0.04$  (volume-law)",
    r"$p = 0.16$  (near-critical)",
    r"$p = 0.32$  (area-law)",
]
COLORS = ["steelblue", "darkorange", "firebrick"]
MARKERS = ["o", "s", "^"]

SUBSYSTEM_SIZES = list(range(1, L // 2 + 1))


def compute_entropy_profile(L, p, n_samples, n_steps, warmup, seed=0):
    """Disorder-average S(A) over all subsystem sizes for a single (L, p)."""
    ss = np.random.SeedSequence(seed)
    rngs = [np.random.default_rng(c) for c in ss.spawn(n_samples)]
    subsystem_sizes = list(range(1, L // 2 + 1))
    all_entropies = np.zeros((n_samples, len(subsystem_sizes)))

    for k, rng in enumerate(rngs):
        state = run_circuit(L, p, n_steps, warmup=warmup, rng=rng)
        for j, size in enumerate(subsystem_sizes):
            A = list(range(size))
            all_entropies[k, j] = entanglement_entropy(state, A)

    mean_S = np.mean(all_entropies, axis=0)
    sem_S = np.std(all_entropies, axis=0, ddof=1) / np.sqrt(n_samples)
    return mean_S, sem_S


def main():
    print(f"Computing entanglement profiles: L={L}, {N_SAMPLES} samples per p value")
    print(f"Subsystem sizes: 1 to {L // 2}")
    print()

    all_means = []
    all_sems = []

    for idx, p in enumerate(P_VALUES):
        print(f"  p = {p:.2f} ...")
        mean_S, sem_S = compute_entropy_profile(
            L, p, N_SAMPLES, N_STEPS, WARMUP, seed=idx * 100 + 7
        )
        all_means.append(mean_S)
        all_sems.append(sem_S)
        print(f"    S(|A|=1) = {mean_S[0]:.3f}, S(|A|=L/2) = {mean_S[-1]:.3f}")

    all_means = np.array(all_means)
    all_sems = np.array(all_sems)

    os.makedirs(os.path.dirname(DATA_OUT), exist_ok=True)
    np.savez(
        DATA_OUT,
        L=L,
        p_values=np.array(P_VALUES),
        subsystem_sizes=np.array(SUBSYSTEM_SIZES),
        mean_entropy=all_means,
        sem_entropy=all_sems,
        n_samples=N_SAMPLES,
    )
    print(f"Saved data: {DATA_OUT}")

    # ---------------------------------------------------------------------- #
    # Figure
    # ---------------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7, 5))

    half_chain_size = L // 2
    size_arr = np.array(SUBSYSTEM_SIZES)

    for idx, (p, label) in enumerate(zip(P_VALUES, P_LABELS)):
        mean_S = all_means[idx]
        sem_S = all_sems[idx]
        c = COLORS[idx]
        mk = MARKERS[idx]

        ax.errorbar(
            size_arr,
            mean_S,
            yerr=sem_S,
            fmt=mk + "-",
            color=c,
            markersize=6,
            capsize=3,
            linewidth=2.0,
            label=label,
            zorder=3,
        )

    # Add reference lines
    # Volume-law reference: S = size (1 ebit per qubit, full volume law)
    s_ref = np.arange(1, half_chain_size + 1, dtype=float)
    ax.plot(
        size_arr,
        s_ref,
        "k--",
        linewidth=1.2,
        alpha=0.5,
        label=r"$S = |A|$  (pure volume law)",
        zorder=2,
    )

    # Log reference at near-critical: fit S ~ alpha * ln(|A|) + c
    # Use the p=0.16 data for the log fit
    x_log = np.log(size_arr.astype(float))
    y_nc = all_means[1]
    fit = linregress(x_log, y_nc)
    alpha_fit = fit.slope
    y_log_fit = alpha_fit * x_log + fit.intercept
    ax.plot(
        size_arr,
        y_log_fit,
        "--",
        color="darkorange",
        linewidth=1.4,
        alpha=0.6,
        zorder=1,
        label=rf"Log fit: $\alpha_\mathrm{{eff}} = {alpha_fit:.2f}$",
    )

    ax.set_xlabel(r"Subsystem size  $|A|$")
    ax.set_ylabel(r"Entanglement entropy  $S(A)$  [ebits]")
    ax.set_title(
        rf"Entanglement Profile  $S(A)$ vs $|A|$  ($L = {L}$, {N_SAMPLES} samples)"
    )
    ax.set_xlim(0.5, half_chain_size + 0.5)
    ax.set_ylim(bottom=0)
    ax.set_xticks(size_arr)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, linestyle="--", color="lightgray", alpha=0.35)

    # Annotate phase labels
    ax.text(
        1.2,
        all_means[0][-1] + 0.15,
        "Volume law",
        fontsize=10,
        color=COLORS[0],
        style="italic",
    )
    ax.text(
        1.2,
        all_means[2][-1] + 0.08,
        "Area law",
        fontsize=10,
        color=COLORS[2],
        style="italic",
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_PNG}")

    # Print summary
    print("\n--- Entanglement profile summary ---")
    print(f"  System size L = {L}, {N_SAMPLES} disorder realizations per p value")
    print(f"  Subsystem A = first |A| qubits [0, ..., |A|-1]")
    print()
    for idx, p in enumerate(P_VALUES):
        s_half = all_means[idx][half_chain_size - 1]
        s_sem_half = all_sems[idx][half_chain_size - 1]
        print(f"  p = {p:.2f}:  S(L/2={half_chain_size}) = {s_half:.3f} +/- {s_sem_half:.3f}")
    print()
    print(f"  Log fit at p=0.16: alpha_eff = {alpha_fit:.3f}")
    print(
        "  (Small-system fit: alpha_eff << 1.6 expected at L=20; see main report)"
    )


if __name__ == "__main__":
    main()
