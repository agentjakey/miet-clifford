"""
Figure 1: Measurement-induced entanglement phase transition.
S(L/2) vs p for multiple system sizes, showing volume-to-area-law crossing.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.simulation import load_results

SWEEP_PATH = os.path.join(os.path.dirname(__file__), "../data/sweep_results.npz")
QUICK_PATH = os.path.join(os.path.dirname(__file__), "../data/quick_results.npz")
OUT_PDF    = os.path.join(os.path.dirname(__file__), "../figures/fig1_phase_diagram.pdf")
OUT_PNG    = os.path.join(os.path.dirname(__file__), "../figures/fig1_phase_diagram.png")


def _load_best():
    """Load sweep data; drop L rows with any NaN; fall back to quick if needed."""
    def filter_valid(L_vals, p_vals, mean, std, sem):
        valid = [i for i, L in enumerate(L_vals)
                 if not np.any(np.isnan(mean[i]))]
        if not valid:
            return None
        idx = np.array(valid)
        return ([L_vals[i] for i in valid],
                p_vals, mean[idx], std[idx], sem[idx])

    # Try production sweep first
    if os.path.exists(SWEEP_PATH):
        L, p, mean, std, sem = load_results(SWEEP_PATH)
        result = filter_valid(L, p, mean, std, sem)
        if result is not None and len(result[0]) >= 1:
            n_valid = len(result[0])
            source  = SWEEP_PATH
            # If the sweep has fewer L values than quick, prefer quick for richer plot
            if n_valid >= 2:
                print(f"Loaded {SWEEP_PATH}  ({n_valid} complete L values)")
                return result, source

    # Fall back to quick results
    L, p, mean, std, sem = load_results(QUICK_PATH)
    result = filter_valid(L, p, mean, std, sem)
    print(f"Loaded {QUICK_PATH}  ({len(result[0])} complete L values)")
    return result, QUICK_PATH


def _find_crossing(p, s1, s2):
    """Find p where two curves cross (interpolated)."""
    diff = s1 - s2
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        return None
    i = sign_changes[0]
    # Linear interpolation
    d0, d1 = diff[i], diff[i + 1]
    p0, p1 = p[i], p[i + 1]
    return float(p0 - d0 * (p1 - p0) / (d1 - d0))


def main():
    plt.rcParams.update({'font.size': 12})

    (L_vals, p_vals, mean_S, std_S, sem_S), source = _load_best()
    n_L = len(L_vals)

    # Color map: darkest for largest L
    cmap   = matplotlib.colormaps.get_cmap("plasma").resampled(n_L + 2)
    colors = [cmap(n_L + 1 - i) for i in range(n_L)]

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, L in enumerate(L_vals):
        c = colors[i]
        ax.plot(p_vals, mean_S[i], color=c, linewidth=2, label=f"$L = {L}$", zorder=3)
        ax.fill_between(p_vals,
                        mean_S[i] - sem_S[i],
                        mean_S[i] + sem_S[i],
                        color=c, alpha=0.25, zorder=2)

    # Phase transition marker
    ax.axvline(0.16, color="dimgray", linestyle="--", linewidth=1.4,
               label=r"$p_c \approx 0.16$", zorder=4)

    # Phase labels
    y_mid  = float(np.nanmax(mean_S)) * 0.55
    y_low  = float(np.nanmax(mean_S)) * 0.12
    ax.text(0.05, y_mid, r"Volume-law  $S \propto L$",
            fontsize=11, color="#1A5276",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#D6EAF8",
                      edgecolor="#1A5276", alpha=0.8))
    ax.text(0.32, y_low, r"Area-law  $S = \mathcal{O}(1)$",
            fontsize=11, color="#7B241C",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FADBD8",
                      edgecolor="#7B241C", alpha=0.8))

    # Axes
    ax.set_xlabel(r"Measurement rate  $p$", fontsize=13)
    ax.set_ylabel(r"Half-chain entanglement entropy  $S(L/2)$  [ebits]",
                  fontsize=12)
    ax.set_title("Measurement-Induced Entanglement Phase Transition", fontsize=13)
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(bottom=0)
    ax.set_xticks(np.arange(0.0, 0.55, 0.1))
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.grid(True, which="major", linestyle="--", color="lightgray", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_PNG}")

    # ── Analytics ────────────────────────────────────────────────────────────
    print("\n--- p_c estimates (max |dS/dp|, midpoint of steepest interval) ---")
    for i, L in enumerate(L_vals):
        ds   = np.abs(np.diff(mean_S[i]))
        dp   = np.diff(p_vals)
        grad = ds / dp
        idx  = int(np.argmax(grad))
        p_c  = float((p_vals[idx] + p_vals[idx + 1]) / 2.0)
        print(f"  L={L:>3}:  p_c ~ {p_c:.3f}  (|dS/dp|_max = {grad[idx]:.2f})")

    if n_L >= 2:
        print("\n--- Crossings of adjacent L pairs ---")
        for i in range(n_L - 1):
            p_cross = _find_crossing(p_vals, mean_S[i], mean_S[i + 1])
            if p_cross is not None:
                print(f"  L={L_vals[i]} x L={L_vals[i+1]}: crossing at p ~ {p_cross:.3f}")
            else:
                print(f"  L={L_vals[i]} x L={L_vals[i+1]}: no crossing found in [0, 0.5]")

        # Explicitly report the two largest
        print(f"\n--- Crossing of two largest L curves "
              f"(L={L_vals[-2]}, L={L_vals[-1]}) ---")
        p_cross = _find_crossing(p_vals, mean_S[-2], mean_S[-1])
        if p_cross is not None:
            print(f"  Crossing at p ~ {p_cross:.3f}")
        else:
            print("  No crossing found — more L values or p points needed.")


if __name__ == "__main__":
    main()
