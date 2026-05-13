"""
Figure 2: Log scaling of half-chain entanglement entropy at the MIPT critical point.

Physics note:
  At p_c, S(L/2) = alpha * ln(L/2) + const.  We fit and report alpha directly.
  alpha is the raw log-scaling coefficient (~1.6 per Li et al. 2019 for random
  Clifford circuits).  We do NOT compute c_eff = 6*alpha or call it a CFT central
  charge: the MIPT critical point is described by a replica field theory, not a
  standard unitary CFT (Jian et al. PRB 101, 104302 2020; Bao et al. PRB 101,
  104301 2020).

References:
  Li, Chen, Fisher, PRB 100, 134306 (2019)
  Jian, You, Vasseur, Ludwig, PRB 101, 104302 (2020)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.simulation import load_results

SWEEP_PATH = os.path.join(os.path.dirname(__file__), "../data/sweep_results.npz")
QUICK_PATH = os.path.join(os.path.dirname(__file__), "../data/quick_results.npz")
OUT_PDF    = os.path.join(os.path.dirname(__file__), "../figures/fig2_log_scaling.pdf")
OUT_PNG    = os.path.join(os.path.dirname(__file__), "../figures/fig2_log_scaling.png")


def _load_best():
    """Load data; drop NaN L rows; prefer sweep if >= 2 valid L, else quick."""
    def filter_valid(L_vals, p_vals, mean, std, sem):
        ok = [i for i in range(len(L_vals)) if not np.any(np.isnan(mean[i]))]
        if not ok:
            return None
        idx = np.array(ok)
        return [L_vals[i] for i in ok], p_vals, mean[idx], std[idx], sem[idx]

    if os.path.exists(SWEEP_PATH):
        result = filter_valid(*load_results(SWEEP_PATH))
        if result is not None and len(result[0]) >= 2:
            print(f"Using {SWEEP_PATH}  ({len(result[0])} L values)")
            return result

    result = filter_valid(*load_results(QUICK_PATH))
    print(f"Using {QUICK_PATH}  ({len(result[0])} L values)")
    return result


def main():
    plt.rcParams.update({'font.size': 12})

    L_vals, p_vals, mean_S, std_S, sem_S = _load_best()
    n_L = len(L_vals)

    # Pick p closest to 0.16
    p_idx  = int(np.argmin(np.abs(p_vals - 0.16)))
    p_crit = float(p_vals[p_idx])
    print(f"Using p = {p_crit:.4f} (closest grid point to 0.16)")

    # Log scaling data: x = ln(L/2), y = S(L/2) at p_crit
    x_log  = np.array([np.log(L / 2.0) for L in L_vals])
    y_vals = mean_S[:, p_idx]
    y_errs = sem_S[:, p_idx]

    # Linear fit: S = alpha * ln(L/2) + const
    fit = linregress(x_log, y_vals)
    alpha     = float(fit.slope)
    intercept = float(fit.intercept)
    # stderr is 0 (nan from scipy) when n=2; handle gracefully
    alpha_err = float(fit.stderr) if (fit.stderr is not None and
                                      np.isfinite(fit.stderr) and
                                      n_L > 2) else float('nan')

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4.5))

    # ── LEFT PANEL: S vs p (compact phase diagram) ───────────────────────────
    cmap   = matplotlib.colormaps.get_cmap("plasma").resampled(n_L + 2)
    colors = [cmap(n_L + 1 - i) for i in range(n_L)]

    for i, L in enumerate(L_vals):
        c = colors[i]
        ax_left.plot(p_vals, mean_S[i], color=c, linewidth=2, label=f"$L={L}$", zorder=3)
        ax_left.fill_between(p_vals,
                             mean_S[i] - sem_S[i],
                             mean_S[i] + sem_S[i],
                             color=c, alpha=0.25, zorder=2)

    ax_left.axvspan(0.13, 0.19, color="lightyellow", alpha=0.6, zorder=1,
                    label="Critical region")
    ax_left.text(0.155, float(np.nanmax(mean_S)) * 0.88, "Critical\nregion",
                 ha="center", va="top", fontsize=9, color="goldenrod",
                 fontweight="bold")
    ax_left.axvline(p_crit, color="dimgray", linestyle=":", linewidth=1.2)
    ax_left.set_xlabel(r"Measurement rate  $p$", fontsize=12)
    ax_left.set_ylabel(r"$S(L/2)$  [ebits]", fontsize=12)
    ax_left.set_title("Phase Diagram", fontsize=12)
    ax_left.set_xlim(0.0, 0.5)
    ax_left.set_ylim(bottom=0)
    ax_left.set_xticks(np.arange(0.0, 0.55, 0.1))
    ax_left.legend(fontsize=10, loc="upper right")
    ax_left.grid(True, linestyle="--", color="lightgray", alpha=0.3)

    # ── RIGHT PANEL: S vs ln(L/2) at p_crit ─────────────────────────────────
    x_fit = np.linspace(x_log.min() - 0.1, x_log.max() + 0.1, 200)
    y_fit = alpha * x_fit + intercept

    ax_right.errorbar(x_log, y_vals, yerr=y_errs,
                      fmt="o", color="navy", markersize=8, capsize=4,
                      linewidth=1.5, zorder=4, label="Data")
    ax_right.plot(x_fit, y_fit, "r--", linewidth=1.8,
                  label=r"Fit: $S = \alpha\,\ln(L/2) + c$", zorder=3)

    # L labels next to data points
    for i, L in enumerate(L_vals):
        ax_right.annotate(f"$L={L}$", (x_log[i], y_vals[i]),
                          textcoords="offset points", xytext=(6, 3),
                          fontsize=9, color="navy")

    # Annotation box
    if np.isfinite(alpha_err):
        ann_text = (f"$\\alpha = {alpha:.3f} \\pm {alpha_err:.3f}$\n"
                    f"Li et al. (2019): $\\alpha \\approx 1.6$")
    else:
        ann_text = (f"$\\alpha = {alpha:.3f}$"
                    + (f"  (n={n_L} pts)" if n_L <= 2 else "") +
                    f"\nLi et al. (2019): $\\alpha \\approx 1.6$")

    ax_right.text(0.05, 0.95, ann_text,
                  transform=ax_right.transAxes,
                  fontsize=10, va="top",
                  bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Disclaimer note below annotation
    ax_right.text(0.05, 0.60,
                  r"$\alpha$ is the raw log-scaling coefficient" + "\n"
                  r"at the MIPT critical point." + "\n"
                  r"It is NOT a CFT central charge." + "\n"
                  "See Jian et al. (2020).",
                  transform=ax_right.transAxes,
                  fontsize=8, va="top", style="italic", color="#444444")

    ax_right.set_xlabel(r"$\ln(L/2)$", fontsize=12)
    ax_right.set_ylabel(r"$S(L/2)$  [ebits]", fontsize=12)
    ax_right.set_title(f"Log Scaling at $p = {p_crit:.3f} \\approx p_c$",
                       fontsize=12)
    ax_right.legend(fontsize=10, loc="lower right")
    ax_right.grid(True, linestyle="--", color="lightgray", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_PNG}")

    # ── Analytics printout ────────────────────────────────────────────────────
    print(f"\n--- Log scaling fit at p = {p_crit:.4f} ---")
    print(f"  alpha          = {alpha:.4f}"
          + (f" +/- {alpha_err:.4f}" if np.isfinite(alpha_err) else "  (n<=2: no stderr)"))
    print(f"  Li et al. 2019   alpha ~ 1.6  (random Clifford circuits)")
    if np.isfinite(alpha) and alpha > 0:
        ratio = alpha / 1.6
        print(f"  Ratio alpha/1.6  = {ratio:.3f}  "
              + ("(consistent)" if 0.5 < ratio < 2.0 else "(large deviation -- more data needed)"))
    if n_L <= 2:
        print(f"\n  NOTE: only {n_L} L value(s) available -- fit is under-constrained.")
        print("  Run scripts/run_sweep.py to completion for a reliable alpha estimate.")


if __name__ == "__main__":
    main()
