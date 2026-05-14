"""
Figure 3: Finite-size scaling collapse for the MIPT.

Simplified ansatz (Jian, You, Vasseur, Ludwig, PRB 101, 104302, 2020):
    S(L, p) = f( (p - p_c) * L^{1/nu} )

IMPORTANT LIMITATIONS
---------------------
(1) The simplified ansatz omits the additive logarithmic term that is present
    at criticality:
        S(L, p_c) = alpha_eff * ln(L/2) + const
    Fitting the raw entropy with the simplified form absorbs this logarithmic
    term into systematically shifted pc_eff and nu_eff.  The collapse is
    therefore DIAGNOSTIC only, not a precision measurement.

(2) At L <= 24, finite-size corrections dominate all extracted parameters.
    pc_eff and nu_eff are finite-size effective estimates.  Literature values
    (Li 2019, Zabalo 2020) were obtained at L up to ~512.

(3) Bootstrap uncertainties quantify statistical precision of the estimator
    given the available data.  They do not capture systematic bias from
    finite-size corrections, which is the dominant source of deviation from
    thermodynamic values.

Literature: p_c ~ 0.16, nu ~ 1.28-1.3  (Li et al. 2019; Zabalo et al. 2020).
"""
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d

mpl.rcParams.update({
    'font.size':        12,
    'axes.labelsize':   13,
    'axes.titlesize':   13,
    'xtick.labelsize':  11,
    'ytick.labelsize':  11,
    'legend.fontsize':  11,
    'figure.dpi':       300,
})

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.simulation import load_results

SWEEP_PATH = os.path.join(os.path.dirname(__file__), "../data/sweep_results.npz")
QUICK_PATH = os.path.join(os.path.dirname(__file__), "../data/quick_results.npz")
OUT_PDF    = os.path.join(os.path.dirname(__file__), "../figures/fig3_scaling_collapse.pdf")
OUT_PNG    = os.path.join(os.path.dirname(__file__), "../figures/fig3_scaling_collapse.png")
OUT_SENS   = os.path.join(os.path.dirname(__file__), "../data/fss_sensitivity.txt")
OUT_CFG    = os.path.join(os.path.dirname(__file__), "../data/fss_config.json")

MARKERS    = ["o", "s", "^", "D", "v", "P"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1, 1, 1))]

# Sensitivity thresholds: warn if estimates vary more than these amounts
# across the sensitivity variants.
_WARN_PC_RANGE  = 0.03
_WARN_NU_RANGE  = 0.30


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


def _collapse_cost(params, L_vals, p_vals, mean_S,
                   alpha_eff=0.0, p_lo=None, p_hi=None):
    """Mean squared deviation between adjacent-L collapse curves.

    The simplified ansatz S(L,p) = f((p-pc)*L^{1/nu}) omits the additive
    logarithmic correction alpha_eff*ln(L/2) present at p_c.  When alpha_eff
    is nonzero (experimental mode), the correction is subtracted before
    computing the collapse.  This does not fix the systematic bias -- it
    shifts it -- and should be treated as diagnostic only.

    p_lo, p_hi: restrict the p range included in the cost (default: full range).
    """
    pc_eff, nu_eff = params
    if nu_eff <= 0:
        return 1e9

    # Optional log subtraction (experimental/diagnostic).
    if alpha_eff != 0.0:
        correction = np.array([alpha_eff * np.log(L / 2.0) for L in L_vals])
        S_work = mean_S - correction[:, None]
    else:
        S_work = mean_S

    # Optionally restrict to a p window around the expected crossing.
    if p_lo is not None or p_hi is not None:
        lo   = p_lo if p_lo is not None else p_vals[0]
        hi   = p_hi if p_hi is not None else p_vals[-1]
        pmask = (p_vals >= lo) & (p_vals <= hi)
        p_use = p_vals[pmask]
        S_use = S_work[:, pmask]
    else:
        p_use = p_vals
        S_use = S_work

    curves = []
    for i, L in enumerate(L_vals):
        x     = (p_use - pc_eff) * (L ** (1.0 / nu_eff))
        order = np.argsort(x)
        curves.append((x[order], S_use[i][order]))

    x_lo = max(c[0][0]  for c in curves)
    x_hi = min(c[0][-1] for c in curves)
    if x_lo >= x_hi:
        return 1e9

    x_grid  = np.linspace(x_lo, x_hi, 60)
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

    cost, n_pairs = 0.0, 0
    for k in range(len(interps) - 1):
        cost += float(np.mean((interps[k] - interps[k + 1]) ** 2))
        n_pairs += 1
    return cost / max(n_pairs, 1)


def _optimize(L_vals, p_vals, mean_S, p_c0=0.16, nu0=1.3,
              alpha_eff=0.0, p_lo=None, p_hi=None):
    bounds_pc = (0.08, 0.28)
    bounds_nu = (0.5, 3.0)

    def cost_bounded(params):
        pc, nu = params
        if not (bounds_pc[0] <= pc <= bounds_pc[1]):
            return 1e9
        if not (bounds_nu[0] <= nu <= bounds_nu[1]):
            return 1e9
        return _collapse_cost(params, L_vals, p_vals, mean_S,
                              alpha_eff=alpha_eff, p_lo=p_lo, p_hi=p_hi)

    res = minimize(cost_bounded, x0=[p_c0, nu0],
                   method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 4000})
    return float(res.x[0]), float(res.x[1])


def _bootstrap(L_vals, p_vals, mean_S, sem_S, n_boot=100,
               p_c0=0.16, nu0=1.3, alpha_eff=0.0, p_lo=None, p_hi=None):
    pc_samples = np.zeros(n_boot)
    nu_samples = np.zeros(n_boot)
    for b in range(n_boot):
        noise  = np.random.randn(*mean_S.shape) * sem_S
        S_boot = mean_S + noise
        pc_b, nu_b = _optimize(L_vals, p_vals, S_boot,
                               p_c0=p_c0, nu0=nu0,
                               alpha_eff=alpha_eff, p_lo=p_lo, p_hi=p_hi)
        pc_samples[b] = pc_b
        nu_samples[b] = nu_b
    return pc_samples, nu_samples


def _sensitivity_analysis(L_vals, p_vals, mean_S, pc0, nu0):
    """Run FSS fits under different L subsets and p windows.

    Returns a list of dicts with keys:
        label, L_used, p_window, pc_eff, nu_eff
    """
    rows = []

    def _fit(L_sub, p_lo=None, p_hi=None, label="", p_window="full"):
        if len(L_sub) < 2:
            return None
        idx   = [L_vals.index(L) for L in L_sub]
        m_sub = mean_S[np.array(idx)]
        try:
            pc, nu = _optimize(L_sub, p_vals, m_sub,
                               p_c0=pc0, nu0=nu0, p_lo=p_lo, p_hi=p_hi)
        except Exception:
            pc, nu = float('nan'), float('nan')
        return {'label': label, 'L_used': L_sub,
                'p_window': p_window, 'pc_eff': pc, 'nu_eff': nu}

    # --- L subsets (full p range) ---
    L_subsets = [
        ("All L",      L_vals),
        ("L >= 12",    [L for L in L_vals if L >= 12]),
        ("L >= 16",    [L for L in L_vals if L >= 16]),
    ]
    for label, L_sub in L_subsets:
        r = _fit(L_sub, label=label)
        if r:
            rows.append(r)

    # --- p windows (all L) ---
    p_windows = [
        ("p in [0.06, 0.35]", 0.06, 0.35),
        ("p in [0.10, 0.25]", 0.10, 0.25),
    ]
    for pw_label, p_lo, p_hi in p_windows:
        r = _fit(L_vals, p_lo=p_lo, p_hi=p_hi,
                 label=f"All L, {pw_label}",
                 p_window=f"[{p_lo}, {p_hi}]")
        if r:
            rows.append(r)

    return rows


def _format_sensitivity_table(rows, pc_main, nu_main):
    lines = []
    lines.append("=" * 70)
    lines.append("FSS Sensitivity Analysis")
    lines.append("Bootstrap uncertainties NOT included; these are point estimates.")
    lines.append("=" * 70)
    lines.append(f"{'Variant':<32} {'L used':<16} {'p window':<18} "
                 f"{'pc_eff':>8} {'nu_eff':>7}")
    lines.append("-" * 70)
    for r in rows:
        pc_str = f"{r['pc_eff']:.4f}" if not np.isnan(r['pc_eff']) else "  n/a "
        nu_str = f"{r['nu_eff']:.4f}" if not np.isnan(r['nu_eff']) else "  n/a "
        lines.append(f"{r['label']:<32} {str(r['L_used']):<16} "
                     f"{r['p_window']:<18} {pc_str:>8} {nu_str:>7}")
    lines.append("-" * 70)

    valid_pc = [r['pc_eff'] for r in rows if not np.isnan(r['pc_eff'])]
    valid_nu = [r['nu_eff'] for r in rows if not np.isnan(r['nu_eff'])]
    if valid_pc:
        lines.append(f"  pc_eff range: [{min(valid_pc):.4f}, {max(valid_pc):.4f}]  "
                     f"(spread = {max(valid_pc)-min(valid_pc):.4f})")
    if valid_nu:
        lines.append(f"  nu_eff range: [{min(valid_nu):.4f}, {max(valid_nu):.4f}]  "
                     f"(spread = {max(valid_nu)-min(valid_nu):.4f})")
    lines.append("=" * 70)
    return "\n".join(lines)


def _check_sensitivity_warning(rows):
    valid_pc = [r['pc_eff'] for r in rows if not np.isnan(r['pc_eff'])]
    valid_nu = [r['nu_eff'] for r in rows if not np.isnan(r['nu_eff'])]
    warn = False
    if valid_pc and (max(valid_pc) - min(valid_pc)) > _WARN_PC_RANGE:
        warn = True
    if valid_nu and (max(valid_nu) - min(valid_nu)) > _WARN_NU_RANGE:
        warn = True
    return warn


def main():
    L_vals, p_vals, mean_S, std_S, sem_S = _load_best()
    n_L = len(L_vals)

    # ------------------------------------------------------------------ #
    # Main FSS fit
    # ------------------------------------------------------------------ #
    print("Optimising FSS collapse (simplified ansatz, raw entropy) ...")
    pc_eff, nu_eff = _optimize(L_vals, p_vals, mean_S)
    print(f"  pc_eff = {pc_eff:.4f},  nu_eff = {nu_eff:.4f}")
    print("  NOTE: these are finite-size effective estimates at L<=24.")
    print("  The simplified ansatz omits the critical log term; collapse is diagnostic.")

    print("Bootstrap uncertainty (100 resamples) ...")
    pc_boot, nu_boot = _bootstrap(L_vals, p_vals, mean_S, sem_S,
                                  n_boot=100, p_c0=pc_eff, nu0=nu_eff)
    pc_err = float(np.std(pc_boot, ddof=1))
    nu_err = float(np.std(nu_boot, ddof=1))
    print(f"  pc_eff = {pc_eff:.4f} +/- {pc_err:.4f}  (statistical only)")
    print(f"  nu_eff = {nu_eff:.4f} +/- {nu_err:.4f}  (statistical only)")

    # ------------------------------------------------------------------ #
    # Sensitivity analysis
    # ------------------------------------------------------------------ #
    print("\nRunning sensitivity analysis ...")
    sens_rows = _sensitivity_analysis(L_vals, p_vals, mean_S,
                                      pc0=pc_eff, nu0=nu_eff)
    sens_table = _format_sensitivity_table(sens_rows, pc_eff, nu_eff)
    print(sens_table)

    if _check_sensitivity_warning(sens_rows):
        print("\nWARNING: FSS estimates are sensitive to fitting choices and "
              "should be interpreted as finite-size effective values.")

    # Save sensitivity table
    os.makedirs(os.path.dirname(OUT_SENS), exist_ok=True)
    with open(OUT_SENS, 'w') as f:
        f.write(sens_table + "\n")
    print(f"\nSaved sensitivity table: {OUT_SENS}")

    # Save fit configuration
    cfg = {
        'L_values':     L_vals,
        'n_L':          n_L,
        'pc_eff':       pc_eff,
        'pc_err_boot':  pc_err,
        'nu_eff':       nu_eff,
        'nu_err_boot':  nu_err,
        'note_pc':      'finite-size effective; bootstrap statistical error only',
        'note_nu':      'finite-size effective; bootstrap statistical error only',
        'ansatz':       'S(L,p) = f((p-pc)*L^(1/nu)); omits critical log term',
        'collapse_type': 'diagnostic; not a precision measurement',
        'lit_pc':        0.16,
        'lit_nu':        1.28,
        'sensitivity_variants': [
            {k: (v if not isinstance(v, float) or not np.isnan(v) else None)
             for k, v in r.items()}
            for r in sens_rows
        ],
    }
    with open(OUT_CFG, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved fit config:        {OUT_CFG}")

    # ------------------------------------------------------------------ #
    # Figure
    # ------------------------------------------------------------------ #
    cmap   = matplotlib.colormaps.get_cmap("plasma").resampled(n_L + 2)
    colors = [cmap(n_L + 1 - i) for i in range(n_L)]

    fig, (ax_raw, ax_col) = plt.subplots(1, 2, figsize=(12, 5))

    # LEFT: raw S vs p
    for i, L in enumerate(L_vals):
        c  = colors[i]
        ls = LINESTYLES[i % len(LINESTYLES)]
        ax_raw.plot(p_vals, mean_S[i], color=c, linewidth=2, linestyle=ls,
                    label=f"$L={L}$", zorder=3)
        ax_raw.fill_between(p_vals,
                            mean_S[i] - sem_S[i],
                            mean_S[i] + sem_S[i],
                            color=c, alpha=0.25, zorder=2)

    ax_raw.axvline(pc_eff, color="dimgray", linestyle="--", linewidth=1.4,
                   label=f"$p_c^{{\\mathrm{{eff}}}}={pc_eff:.3f}$")
    ax_raw.set_xlabel(r"Measurement rate  $p$")
    ax_raw.set_ylabel(r"$S(L/2)$  [ebits]")
    ax_raw.set_title("Raw Data")
    ax_raw.set_xlim(0.0, 0.5)
    ax_raw.set_ylim(bottom=0)
    ax_raw.set_xticks(np.arange(0.0, 0.55, 0.1))
    ax_raw.legend(loc="upper right")
    ax_raw.grid(True, linestyle="--", color="lightgray", alpha=0.3)

    # RIGHT: collapse -- raw entropy, simplified ansatz
    for i, L in enumerate(L_vals):
        x  = (p_vals - pc_eff) * (L ** (1.0 / nu_eff))
        y  = mean_S[i]
        e  = sem_S[i]
        c  = colors[i]
        mk = MARKERS[i % len(MARKERS)]
        ls = LINESTYLES[i % len(LINESTYLES)]
        ax_col.errorbar(x, y, yerr=e, fmt=mk, color=c, markersize=6,
                        capsize=3, linewidth=2.0, linestyle=ls,
                        label=f"$L={L}$", zorder=3)

    ax_col.set_xlabel(r"$(p - p_c^{\mathrm{eff}})\, L^{1/\nu_{\mathrm{eff}}}$")
    ax_col.set_ylabel(r"$S(L/2)$  [ebits]")
    ax_col.set_title(
        f"FSS Collapse (diagnostic): "
        f"$p_c^{{\\mathrm{{eff}}}}={pc_eff:.3f}$, "
        f"$\\nu_{{\\mathrm{{eff}}}}={nu_eff:.2f}$")
    ax_col.legend(loc="upper right")
    ax_col.grid(True, linestyle="--", color="lightgray", alpha=0.3)

    if n_L >= 3:
        ann = (
            f"Fit (diag.):  $p_c^{{\\mathrm{{eff}}}} = {pc_eff:.3f}"
            f" \\pm {pc_err:.3f}$\n"
            f"             $\\nu_{{\\mathrm{{eff}}}} = {nu_eff:.2f}"
            f" \\pm {nu_err:.2f}$\n"
            r"(stat. only; $L\leq 24$)" + "\n\n"
            "Li et al. (2019):\n"
            r"  $p_c \approx 0.16,\ \nu \approx 1.3$" + "\n\n"
            "Zabalo et al. (2020):\n"
            r"  $p_c \approx 0.160,\ \nu \approx 1.28$"
        )
    else:
        ann = (
            f"Fit ($n_L={n_L}$, diag.):\n"
            f"  $p_c^{{\\mathrm{{eff}}}} = {pc_eff:.3f}$\n"
            f"  $\\nu_{{\\mathrm{{eff}}}} = {nu_eff:.2f}$\n"
            f"  (boot. $\\sigma_{{p_c}}={pc_err:.3f}$,\n"
            f"   $\\sigma_{{\\nu}}={nu_err:.2f}$)\n\n"
            "Li et al. (2019):\n"
            r"  $p_c \approx 0.16,\ \nu \approx 1.3$" + "\n\n"
            "Zabalo et al. (2020):\n"
            r"  $p_c \approx 0.160,\ \nu \approx 1.28$"
        )

    ax_col.text(0.03, 0.97, ann,
                transform=ax_col.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"\nSaved {OUT_PDF}")
    print(f"Saved {OUT_PNG}")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n--- Finite-size effective FSS parameters (L <= 24) ---")
    print(f"  pc_eff = {pc_eff:.4f} +/- {pc_err:.4f}  "
          f"(stat. only; Li: 0.160, Zabalo: 0.160)")
    print(f"  nu_eff = {nu_eff:.4f} +/- {nu_err:.4f}  "
          f"(stat. only; Li: 1.3,   Zabalo: 1.28)")
    print("  Deviations from literature are expected systematic finite-size "
          "effects, not statistical tension.")
    if n_L < 3:
        print(f"\n  NOTE: only {n_L} L values -- FSS fit is underdetermined.")
        print("  Run scripts/run_sweep.py to completion for reliable exponents.")


if __name__ == "__main__":
    main()
