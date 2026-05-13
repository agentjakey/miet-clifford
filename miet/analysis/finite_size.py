"""
Figure 3: Finite-size scaling collapse for the MIPT.

Ansatz (Jian, You, Vasseur, Ludwig, PRB 101, 104302, 2020):
    S(L, p) = f( (p - p_c) * L^{1/nu} )

where nu is the correlation length exponent.
Literature: p_c ~ 0.16, nu ~ 1.3  (Li et al. 2019; Zabalo et al. 2020).

Cost function: for a candidate (p_c, nu), rescale each L curve onto x-coordinates
(p - p_c)*L^{1/nu}, interpolate all curves onto a shared x-grid, then measure the
mean squared deviation between adjacent-L curve pairs.  Minimising this yields the
collapse that makes all curves fall onto the same universal function f.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import os, sys, warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.simulation import load_results

SWEEP_PATH = os.path.join(os.path.dirname(__file__), "../data/sweep_results.npz")
QUICK_PATH = os.path.join(os.path.dirname(__file__), "../data/quick_results.npz")
OUT_PDF    = os.path.join(os.path.dirname(__file__), "../figures/fig3_scaling_collapse.pdf")
OUT_PNG    = os.path.join(os.path.dirname(__file__), "../figures/fig3_scaling_collapse.png")

MARKERS = ["o", "s", "^", "D", "v", "P"]


def _load_best():
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


def _collapse_cost(params, L_vals, p_vals, mean_S):
    """Mean squared deviation between adjacent rescaled curves (interpolated).

    For each (p_c, nu), compute x_i = (p - p_c)*L_i^{1/nu} for every L_i.
    Find the x-range common to all curves; interpolate; sum squared residuals
    between adjacent L pairs over the shared grid.
    """
    p_c, nu = params
    if nu <= 0:
        return 1e9

    curves = []
    for i, L in enumerate(L_vals):
        x = (p_vals - p_c) * (L ** (1.0 / nu))
        y = mean_S[i]
        # Only keep interior points where the curve is monotone-ish
        order = np.argsort(x)
        curves.append((x[order], y[order]))

    # Shared x range (intersection)
    x_lo = max(c[0][0]  for c in curves)
    x_hi = min(c[0][-1] for c in curves)
    if x_lo >= x_hi:
        return 1e9

    x_grid = np.linspace(x_lo, x_hi, 60)
    interps = []
    for (x_c, y_c) in curves:
        mask = (x_c >= x_lo) & (x_c <= x_hi)
        if mask.sum() < 2:
            return 1e9
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = interp1d(x_c[mask], y_c[mask], kind="linear",
                         fill_value="extrapolate")
        interps.append(f(x_grid))

    cost = 0.0
    n_pairs = 0
    for k in range(len(interps) - 1):
        diff = interps[k] - interps[k + 1]
        cost += float(np.mean(diff ** 2))
        n_pairs += 1
    return cost / max(n_pairs, 1)


def _optimize(L_vals, p_vals, mean_S, p_c0=0.16, nu0=1.3):
    bounds_pc = (0.08, 0.28)
    bounds_nu = (0.5, 3.0)

    def cost_bounded(params):
        p_c, nu = params
        if not (bounds_pc[0] <= p_c <= bounds_pc[1]):
            return 1e9
        if not (bounds_nu[0] <= nu <= bounds_nu[1]):
            return 1e9
        return _collapse_cost(params, L_vals, p_vals, mean_S)

    res = minimize(cost_bounded, x0=[p_c0, nu0],
                   method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 4000})
    return float(res.x[0]), float(res.x[1])


def _bootstrap(L_vals, p_vals, mean_S, sem_S, n_boot=100, p_c0=0.16, nu0=1.3):
    pc_samples  = np.zeros(n_boot)
    nu_samples  = np.zeros(n_boot)
    for b in range(n_boot):
        noise = np.random.randn(*mean_S.shape) * sem_S
        S_boot = mean_S + noise
        pc_b, nu_b = _optimize(L_vals, p_vals, S_boot, p_c0=p_c0, nu0=nu0)
        pc_samples[b] = pc_b
        nu_samples[b] = nu_b
    return pc_samples, nu_samples


def main():
    plt.rcParams.update({"font.size": 12})

    L_vals, p_vals, mean_S, std_S, sem_S = _load_best()
    n_L = len(L_vals)

    # ── Optimise collapse ────────────────────────────────────────────────────
    print("Optimising FSS collapse ...")
    p_c_opt, nu_opt = _optimize(L_vals, p_vals, mean_S)
    print(f"  p_c = {p_c_opt:.4f},  nu = {nu_opt:.4f}")

    print(f"Bootstrap uncertainty (100 resamples) ...")
    pc_boot, nu_boot = _bootstrap(L_vals, p_vals, mean_S, sem_S,
                                  n_boot=100, p_c0=p_c_opt, nu0=nu_opt)
    pc_err = float(np.std(pc_boot, ddof=1))
    nu_err = float(np.std(nu_boot, ddof=1))

    # ── Color scheme (match Fig 1) ────────────────────────────────────────────
    cmap   = matplotlib.colormaps.get_cmap("plasma").resampled(n_L + 2)
    colors = [cmap(n_L + 1 - i) for i in range(n_L)]

    fig, (ax_raw, ax_col) = plt.subplots(1, 2, figsize=(12, 5))

    # ── LEFT: raw S vs p ─────────────────────────────────────────────────────
    for i, L in enumerate(L_vals):
        c = colors[i]
        ax_raw.plot(p_vals, mean_S[i], color=c, linewidth=2,
                    label=f"$L={L}$", zorder=3)
        ax_raw.fill_between(p_vals,
                            mean_S[i] - sem_S[i],
                            mean_S[i] + sem_S[i],
                            color=c, alpha=0.25, zorder=2)

    ax_raw.axvline(p_c_opt, color="dimgray", linestyle="--",
                   linewidth=1.4, label=f"$p_c={p_c_opt:.3f}$")
    ax_raw.set_xlabel(r"Measurement rate  $p$", fontsize=12)
    ax_raw.set_ylabel(r"$S(L/2)$  [ebits]", fontsize=12)
    ax_raw.set_title("Raw Data", fontsize=12)
    ax_raw.set_xlim(0.0, 0.5)
    ax_raw.set_ylim(bottom=0)
    ax_raw.set_xticks(np.arange(0.0, 0.55, 0.1))
    ax_raw.legend(fontsize=10, loc="upper right")
    ax_raw.grid(True, linestyle="--", color="lightgray", alpha=0.3)

    # ── RIGHT: collapse plot ─────────────────────────────────────────────────
    for i, L in enumerate(L_vals):
        x = (p_vals - p_c_opt) * (L ** (1.0 / nu_opt))
        y = mean_S[i]
        e = sem_S[i]
        c = colors[i]
        mk = MARKERS[i % len(MARKERS)]
        ax_col.errorbar(x, y, yerr=e, fmt=mk, color=c, markersize=6,
                        capsize=3, linewidth=1.4, label=f"$L={L}$", zorder=3)

    ax_col.set_xlabel(r"$(p - p_c)\, L^{1/\nu}$", fontsize=13)
    ax_col.set_ylabel(r"$S(L/2)$  [ebits]", fontsize=12)
    ax_col.set_title(
        f"FSS Collapse: $p_c = {p_c_opt:.3f}$, $\\nu = {nu_opt:.2f}$",
        fontsize=12)
    ax_col.legend(fontsize=10, loc="upper right")
    ax_col.grid(True, linestyle="--", color="lightgray", alpha=0.3)

    # Annotation: fit results vs literature
    if n_L >= 3:
        ann = (f"Fit:  $p_c = {p_c_opt:.3f} \\pm {pc_err:.3f}$\n"
               f"      $\\nu = {nu_opt:.2f} \\pm {nu_err:.2f}$\n\n"
               "Li et al. (2019):\n"
               r"  $p_c \approx 0.16,\ \nu \approx 1.3$" + "\n\n"
               "Zabalo et al. (2020):\n"
               r"  $p_c \approx 0.160,\ \nu \approx 1.28$")
    else:
        ann = (f"Fit ($n_L={n_L}$):  $p_c = {p_c_opt:.3f}$\n"
               f"              $\\nu = {nu_opt:.2f}$\n"
               f"(bootstrap $\\sigma_{{p_c}}={pc_err:.3f}$,\n"
               f" $\\sigma_{{\\nu}}={nu_err:.2f}$)\n\n"
               "Li et al. (2019):\n"
               r"  $p_c \approx 0.16,\ \nu \approx 1.3$" + "\n\n"
               "Zabalo et al. (2020):\n"
               r"  $p_c \approx 0.160,\ \nu \approx 1.28$")

    ax_col.text(0.03, 0.97, ann,
                transform=ax_col.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_PNG}")

    # ── Analytics printout ───────────────────────────────────────────────────
    print("\n--- Finite-size scaling fit ---")
    print(f"  p_c  = {p_c_opt:.4f} +/- {pc_err:.4f}  "
          f"(Li et al.: 0.160, Zabalo et al.: 0.160)")
    print(f"  nu   = {nu_opt:.4f} +/- {nu_err:.4f}  "
          f"(Li et al.: 1.3,   Zabalo et al.: 1.28)")
    if n_L < 3:
        print(f"\n  NOTE: only {n_L} L values -- FSS fit is underdetermined.")
        print("  Run scripts/run_sweep.py to completion for reliable exponents.")
    else:
        print(f"\n  p_c deviation from Li et al.:  "
              f"{abs(p_c_opt-0.16)/pc_err:.1f} sigma" if pc_err > 0 else "")
        print(f"  nu  deviation from Li et al.:  "
              f"{abs(nu_opt-1.3)/nu_err:.1f} sigma" if nu_err > 0 else "")


if __name__ == "__main__":
    main()
