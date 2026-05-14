"""
Figure 2: Log-scaling of half-chain entropy near the MIPT critical point.

Physics note:
  At p_c, S(L/2) = alpha * ln(L/2) + const.  We fit and report alpha_eff
  directly.  alpha is the raw log-scaling coefficient (~1.6 per Li et al.
  2019 for large random Clifford circuits at L >> 100).  We do NOT compute
  c_eff = 6*alpha or call it a CFT central charge: the MIPT critical point is
  described by a replica field theory, not a standard unitary CFT
  (Jian et al. PRB 101, 104302 2020).

LIMITATIONS:
  The fit uses only five system sizes (L=8,12,16,20,24).  At these sizes the
  system has not entered the asymptotic critical log-scaling regime.  The fit
  result is sensitive to the chosen evaluation p: evaluating at p=0.16 (the
  thermodynamic p_c) places the system in the volume-law side of the
  finite-size crossover, suppressing the apparent slope.  This analysis
  computes alpha_eff at multiple p values and reports the variation as a
  systematic sensitivity.

References:
  Li, Chen, Fisher, PRB 100, 134306 (2019)
  Jian, You, Vasseur, Ludwig, PRB 101, 104302 (2020)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import os, sys

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

SWEEP_PATH  = os.path.join(os.path.dirname(__file__), "../data/sweep_results.npz")
QUICK_PATH  = os.path.join(os.path.dirname(__file__), "../data/quick_results.npz")
OUT_PDF     = os.path.join(os.path.dirname(__file__), "../figures/fig2_log_scaling.pdf")
OUT_PNG     = os.path.join(os.path.dirname(__file__), "../figures/fig2_log_scaling.png")
OUT_SUPP_PDF = os.path.join(os.path.dirname(__file__), "../figures/fig2b_alpha_vs_p.pdf")
OUT_SUPP_PNG = os.path.join(os.path.dirname(__file__), "../figures/fig2b_alpha_vs_p.png")
OUT_TABLE   = os.path.join(os.path.dirname(__file__), "../data/alpha_vs_p.txt")

LINESTYLES  = ["-", "--", "-.", ":", (0, (3, 1, 1, 1, 1, 1))]

# Warn if alpha_eff changes by more than this across nearby p values.
_WARN_ALPHA_RANGE = 0.5


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


def _fit_alpha_at_p(p_target, p_vals, L_vals, mean_S, sem_S):
    """Fit S = alpha*ln(L/2) + const at the p grid point closest to p_target.

    Returns dict with keys: p_used, alpha_eff, alpha_err, r2, intercept,
    L_values_used.
    """
    p_idx  = int(np.argmin(np.abs(p_vals - p_target)))
    p_used = float(p_vals[p_idx])
    x      = np.array([np.log(L / 2.0) for L in L_vals])
    y      = mean_S[:, p_idx]

    fit = linregress(x, y)
    alpha     = float(fit.slope)
    intercept = float(fit.intercept)
    r2        = float(fit.rvalue ** 2)
    n         = len(L_vals)
    alpha_err = (float(fit.stderr)
                 if (fit.stderr is not None and np.isfinite(fit.stderr) and n > 2)
                 else float('nan'))

    return {
        'p_used':       p_used,
        'alpha_eff':    alpha,
        'alpha_err':    alpha_err,
        'r2':           r2,
        'intercept':    intercept,
        'n_sizes':      n,
        'L_values':     list(L_vals),
    }


def _build_alpha_table(p_vals, L_vals, mean_S, sem_S,
                       pc_lit=0.16, pc_cross=None, pc_fss=None):
    """Compute alpha_eff at multiple p values spanning the critical region.

    Evaluation points:
      - All p grid points in [0.10, 0.30] (dense scan)
      - p = pc_lit = 0.16
      - p near pc_cross (crossing estimate) if provided
      - p near pc_fss (FSS estimate) if provided
    """
    # Dense scan: p in [0.10, 0.30]
    scan_mask = (p_vals >= 0.10) & (p_vals <= 0.30)
    scan_ps   = p_vals[scan_mask]

    # Named evaluation points (deduplicated against scan grid)
    named_ps = [pc_lit]
    if pc_cross is not None:
        named_ps.append(float(pc_cross))
    if pc_fss is not None:
        named_ps.append(float(pc_fss))

    all_p_targets = sorted(set(list(scan_ps) + named_ps))

    rows = []
    seen = set()
    for pt in all_p_targets:
        r = _fit_alpha_at_p(pt, p_vals, L_vals, mean_S, sem_S)
        key = round(r['p_used'], 6)
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)

    rows.sort(key=lambda r: r['p_used'])
    return rows


def _format_alpha_table(rows):
    lines = []
    lines.append("=" * 80)
    lines.append("alpha_eff vs p  (all fits: S = alpha_eff * ln(L/2) + const)")
    lines.append("Sizes used: all available L values (see L_values column)")
    lines.append("alpha_err is OLS stderr; meaningful only for n_sizes >= 3")
    lines.append("=" * 80)
    lines.append(f"{'p_used':>8}  {'alpha_eff':>10}  {'alpha_err':>10}  "
                 f"{'R^2':>6}  {'intercept':>10}  {'n_sizes':>7}")
    lines.append("-" * 80)
    for r in rows:
        ae = f"{r['alpha_err']:.4f}" if np.isfinite(r['alpha_err']) else "  n/a  "
        lines.append(f"{r['p_used']:8.4f}  {r['alpha_eff']:10.4f}  {ae:>10}  "
                     f"{r['r2']:6.4f}  {r['intercept']:10.4f}  {r['n_sizes']:7d}")
    lines.append("=" * 80)
    return "\n".join(lines)


def _sensitivity_warning(rows):
    alphas = [r['alpha_eff'] for r in rows if np.isfinite(r['alpha_eff'])]
    if not alphas:
        return False
    return (max(alphas) - min(alphas)) > _WARN_ALPHA_RANGE


def main():
    L_vals, p_vals, mean_S, std_S, sem_S = _load_best()
    n_L = len(L_vals)

    # ------------------------------------------------------------------ #
    # Try to read FSS estimates from saved config (if available)
    # ------------------------------------------------------------------ #
    cfg_path = os.path.join(os.path.dirname(__file__), "../data/fss_config.json")
    pc_fss    = None
    pc_cross  = None
    if os.path.exists(cfg_path):
        try:
            import json
            with open(cfg_path) as f:
                cfg = json.load(f)
            pc_fss = cfg.get('pc_eff')
            print(f"Read pc_eff = {pc_fss:.4f} from {cfg_path}")
        except Exception:
            pass

    cross_path = os.path.join(os.path.dirname(__file__), "../data/crossing_table.txt")
    if os.path.exists(cross_path):
        # Try to parse the mean crossing from the crossing table header
        try:
            with open(cross_path) as f:
                for line in f:
                    if 'mean crossing' in line.lower():
                        pc_cross = float(line.split('=')[-1].strip().split()[0])
                        print(f"Read pc_cross = {pc_cross:.4f} from {cross_path}")
                        break
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Fit at primary point (p=0.16) for the main figure
    # ------------------------------------------------------------------ #
    primary = _fit_alpha_at_p(0.16, p_vals, L_vals, mean_S, sem_S)
    alpha     = primary['alpha_eff']
    alpha_err = primary['alpha_err']
    p_crit    = primary['p_used']
    intercept = primary['intercept']
    print(f"\nPrimary fit at p = {p_crit:.4f}:")
    print(f"  alpha_eff = {alpha:.4f}"
          + (f" +/- {alpha_err:.4f}" if np.isfinite(alpha_err) else "  (n<=2)"))
    print(f"  R^2       = {primary['r2']:.4f}")
    print(f"  NOTE: alpha_eff << 1.6 is expected at L<=24 (see module docstring)")

    # ------------------------------------------------------------------ #
    # Build multi-p alpha table
    # ------------------------------------------------------------------ #
    print("\nComputing alpha_eff at multiple p values ...")
    table_rows = _build_alpha_table(p_vals, L_vals, mean_S, sem_S,
                                    pc_lit=0.16,
                                    pc_cross=pc_cross,
                                    pc_fss=pc_fss)

    table_str = _format_alpha_table(table_rows)
    print(table_str)

    if _sensitivity_warning(table_rows):
        print("\nWARNING: alpha_eff is highly sensitive to the chosen p value. "
              "The fit is underpowered and should not be interpreted as a "
              "measurement of the thermodynamic exponent alpha(p_c).")

    os.makedirs(os.path.dirname(OUT_TABLE), exist_ok=True)
    with open(OUT_TABLE, 'w') as f:
        f.write(table_str + "\n")
    print(f"\nSaved alpha table: {OUT_TABLE}")

    # ------------------------------------------------------------------ #
    # Main figure (two panels)
    # ------------------------------------------------------------------ #
    cmap   = matplotlib.colormaps.get_cmap("plasma").resampled(n_L + 2)
    colors = [cmap(n_L + 1 - i) for i in range(n_L)]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4.5))

    # LEFT: S vs p
    for i, L in enumerate(L_vals):
        c  = colors[i]
        ls = LINESTYLES[i % len(LINESTYLES)]
        ax_left.plot(p_vals, mean_S[i], color=c, linewidth=2, linestyle=ls,
                     label=f"$L={L}$", zorder=3)
        ax_left.fill_between(p_vals,
                             mean_S[i] - sem_S[i],
                             mean_S[i] + sem_S[i],
                             color=c, alpha=0.25, zorder=2)

    ax_left.axvspan(0.13, 0.19, color="lightyellow", alpha=0.6, zorder=1,
                    label="Critical region")
    ax_left.text(0.155, float(np.nanmax(mean_S)) * 0.88, "Critical\nregion",
                 ha="center", va="top", fontsize=10, color="goldenrod",
                 fontweight="bold")
    ax_left.axvline(p_crit, color="dimgray", linestyle=":", linewidth=1.2)
    ax_left.set_xlabel(r"Measurement rate  $p$")
    ax_left.set_ylabel(r"$S(L/2)$  [ebits]")
    ax_left.set_title("Phase Diagram")
    ax_left.set_xlim(0.0, 0.5)
    ax_left.set_ylim(bottom=0)
    ax_left.set_xticks(np.arange(0.0, 0.55, 0.1))
    ax_left.legend(loc="upper right")
    ax_left.grid(True, linestyle="--", color="lightgray", alpha=0.3)

    # RIGHT: S vs ln(L/2) at p_crit
    x_log = np.array([np.log(L / 2.0) for L in L_vals])
    y_vals = mean_S[:, int(np.argmin(np.abs(p_vals - 0.16)))]
    y_errs = sem_S[:, int(np.argmin(np.abs(p_vals - 0.16)))]

    x_fit = np.linspace(x_log.min() - 0.1, x_log.max() + 0.1, 200)
    y_fit = alpha * x_fit + intercept

    ax_right.errorbar(x_log, y_vals, yerr=y_errs,
                      fmt="o", color="navy", markersize=8, capsize=4,
                      linewidth=2.0, zorder=4, label="Data")
    ax_right.plot(x_fit, y_fit, "r--", linewidth=2.0,
                  label=r"Fit: $S = \alpha_{\mathrm{eff}}\,\ln(L/2) + c$",
                  zorder=3)

    for i, L in enumerate(L_vals):
        ax_right.annotate(f"$L={L}$", (x_log[i], y_vals[i]),
                          textcoords="offset points", xytext=(6, 3),
                          fontsize=10, color="navy")

    if np.isfinite(alpha_err):
        ann_text = (
            f"$\\alpha_{{\\mathrm{{eff}}}} = {alpha:.3f} \\pm {alpha_err:.3f}$\n"
            f"(small-system fit; $L\\leq 24$)\n"
            f"Li et al. (2019): $\\alpha \\approx 1.6$\n"
            r"($L \gg 100$; not comparable)"
        )
    else:
        ann_text = (
            f"$\\alpha_{{\\mathrm{{eff}}}} = {alpha:.3f}$"
            + (f"  ($n={n_L}$ pts)" if n_L <= 2 else "") + "\n"
            f"Li et al. (2019): $\\alpha \\approx 1.6$"
        )

    ax_right.text(0.05, 0.97, ann_text,
                  transform=ax_right.transAxes,
                  fontsize=10, va="top",
                  bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax_right.text(0.05, 0.52,
                  r"$\alpha_{\mathrm{eff}}$ is a small-system" + "\n"
                  r"fit coefficient, not a thermodynamic" + "\n"
                  r"exponent.  Not a CFT central charge." + "\n"
                  "See Jian et al. (2020).",
                  transform=ax_right.transAxes,
                  fontsize=9, va="top", style="italic", color="#444444")

    ax_right.set_xlabel(r"$\ln(L/2)$")
    ax_right.set_ylabel(r"$S(L/2)$  [ebits]")
    ax_right.set_title(
        f"Log Scaling at $p = {p_crit:.3f}$  "
        f"($\\alpha_{{\\mathrm{{eff}}}}={alpha:.3f}$)")
    ax_right.legend(loc="lower right")
    ax_right.grid(True, linestyle="--", color="lightgray", alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_PNG}")

    # ------------------------------------------------------------------ #
    # Supplemental figure: alpha_eff vs p
    # ------------------------------------------------------------------ #
    scan_ps    = np.array([r['p_used']     for r in table_rows])
    scan_alpha = np.array([r['alpha_eff']  for r in table_rows])
    scan_err   = np.array([r['alpha_err']  for r in table_rows])

    fig2, ax_s = plt.subplots(figsize=(6, 4))
    has_err = np.isfinite(scan_err)
    if has_err.any():
        ax_s.errorbar(scan_ps[has_err], scan_alpha[has_err], yerr=scan_err[has_err],
                      fmt="o-", color="steelblue", markersize=5, capsize=3,
                      linewidth=1.5, label=r"$\alpha_{\mathrm{eff}}(p)$")
    if (~has_err).any():
        ax_s.plot(scan_ps[~has_err], scan_alpha[~has_err], "o",
                  color="steelblue", markersize=5)

    ax_s.axhline(1.6, color="tomato", linestyle="--", linewidth=1.5,
                 label=r"Li et al. $\alpha \approx 1.6$ ($L\gg 100$)")
    ax_s.axvline(0.16, color="gray", linestyle=":", linewidth=1.2,
                 label=r"$p_c^{\mathrm{lit}} = 0.16$")

    if pc_cross is not None:
        ax_s.axvline(pc_cross, color="darkorange", linestyle="-.", linewidth=1.0,
                     label=f"$p_c^{{\\mathrm{{cross}}}} = {pc_cross:.3f}$")
    if pc_fss is not None:
        ax_s.axvline(pc_fss, color="purple", linestyle="-.", linewidth=1.0,
                     label=f"$p_c^{{\\mathrm{{FSS}}}} = {pc_fss:.3f}$")

    ax_s.set_xlabel(r"Measurement rate $p$")
    ax_s.set_ylabel(r"$\alpha_{\mathrm{eff}}(p)$")
    ax_s.set_title(
        r"$\alpha_{\mathrm{eff}}$ sensitivity to evaluation $p$  "
        f"(L $\\leq$ {max(L_vals)})")
    ax_s.legend(loc="upper right", fontsize=9)
    ax_s.grid(True, linestyle="--", color="lightgray", alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(OUT_SUPP_PDF, dpi=300, bbox_inches="tight")
    fig2.savefig(OUT_SUPP_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_SUPP_PDF}")
    print(f"Saved {OUT_SUPP_PNG}")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print(f"\n--- alpha_eff summary (primary fit at p = {p_crit:.4f}) ---")
    print(f"  alpha_eff  = {alpha:.4f}"
          + (f" +/- {alpha_err:.4f}" if np.isfinite(alpha_err) else "  (n<=2)"))
    print(f"  R^2        = {primary['r2']:.4f}")
    print(f"  Lit. value = ~1.6  (Li 2019, L >> 100)")
    print(f"  The fit uses {n_L} system sizes; it is underpowered and p-sensitive.")
    if n_L <= 2:
        print(f"\n  NOTE: only {n_L} L values -- fit is under-constrained.")
        print("  Run scripts/run_sweep.py for a more reliable estimate.")


if __name__ == "__main__":
    main()
