"""
Figure 1: Measurement-induced entanglement phase transition.

S(L/2) vs p for multiple system sizes, showing the volume-to-area-law crossover.

Adjacent-size crossings are reported as pcross_eff -- finite-size effective
estimates of the critical rate.  The spread of crossings across different L pairs
is a systematic finite-size signal: as L increases, crossings drift toward the
thermodynamic p_c.  This spread is NOT a statistical error and is not reduced by
collecting more disorder realizations at fixed L.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

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

SWEEP_PATH     = os.path.join(os.path.dirname(__file__), "../data/sweep_results.npz")
QUICK_PATH     = os.path.join(os.path.dirname(__file__), "../data/quick_results.npz")
OUT_PDF        = os.path.join(os.path.dirname(__file__), "../figures/fig1_phase_diagram.pdf")
OUT_PNG        = os.path.join(os.path.dirname(__file__), "../figures/fig1_phase_diagram.png")
OUT_CROSS_TXT  = os.path.join(os.path.dirname(__file__), "../data/crossing_table.txt")
OUT_CROSS_JSON = os.path.join(os.path.dirname(__file__), "../data/crossing_table.json")

LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1, 1, 1))]

# Bootstrap resamples for crossing uncertainty
N_BOOT_CROSS = 200


def _load_best():
    def filter_valid(L_vals, p_vals, mean, std, sem):
        valid = [i for i, L in enumerate(L_vals)
                 if not np.any(np.isnan(mean[i]))]
        if not valid:
            return None
        idx = np.array(valid)
        return ([L_vals[i] for i in valid],
                p_vals, mean[idx], std[idx], sem[idx])

    if os.path.exists(SWEEP_PATH):
        L, p, mean, std, sem = load_results(SWEEP_PATH)
        result = filter_valid(L, p, mean, std, sem)
        if result is not None and len(result[0]) >= 2:
            print(f"Loaded {SWEEP_PATH}  ({len(result[0])} complete L values)")
            return result, SWEEP_PATH

    L, p, mean, std, sem = load_results(QUICK_PATH)
    result = filter_valid(L, p, mean, std, sem)
    print(f"Loaded {QUICK_PATH}  ({len(result[0])} complete L values)")
    return result, QUICK_PATH


def _find_crossing(p_vals, s1, s2):
    """Linear interpolation crossing of two curves.

    Returns (p_cross, note):
        p_cross: float or None
        note:    string describing quality of crossing
    """
    diff = s1 - s2
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) == 0:
        return None, "no crossing found in p range"

    i    = sign_changes[0]
    d0   = float(diff[i])
    d1   = float(diff[i + 1])
    p0   = float(p_vals[i])
    p1   = float(p_vals[i + 1])
    p_cross = p0 - d0 * (p1 - p0) / (d1 - d0)

    # Quality note: warn if crossing is near the boundary of the p range or
    # if the curves are nearly parallel at the crossing.
    note = ""
    if p_cross < 0.10 or p_cross > 0.45:
        note = "crossing near p boundary -- poorly resolved"
    elif abs(d1 - d0) < 0.05 * max(abs(d0), abs(d1), 1e-9):
        note = "shallow crossing -- curves nearly parallel"
    elif len(sign_changes) > 1:
        note = "multiple crossings -- non-monotonic difference"

    return float(p_cross), note


def _bootstrap_crossing_uncertainty(p_vals, s1, s2, e1, e2,
                                    n_boot=N_BOOT_CROSS, rng_seed=0):
    """Estimate statistical uncertainty on the crossing via bootstrap resampling.

    Adds Gaussian noise scaled to the SEM of each curve and recomputes the
    crossing for each resample.  Returns std of the bootstrap crossing estimates.
    Returns nan if fewer than 10 valid crossings are found.
    """
    rng = np.random.default_rng(rng_seed)
    crossings = []
    for _ in range(n_boot):
        s1_b = s1 + rng.standard_normal(len(s1)) * e1
        s2_b = s2 + rng.standard_normal(len(s2)) * e2
        pc_b, _ = _find_crossing(p_vals, s1_b, s2_b)
        if pc_b is not None and 0.0 < pc_b < 0.5:
            crossings.append(pc_b)

    if len(crossings) < 10:
        return float('nan')
    return float(np.std(crossings, ddof=1))


def _compute_crossing_table(L_vals, p_vals, mean_S, sem_S):
    """Compute crossing for each adjacent (L_i, L_{i+1}) pair.

    Returns list of dicts with keys:
        L1, L2, pcross_eff, uncertainty, note
    """
    rows = []
    for i in range(len(L_vals) - 1):
        L1 = L_vals[i]
        L2 = L_vals[i + 1]
        s1, s2 = mean_S[i], mean_S[i + 1]
        e1, e2 = sem_S[i],  sem_S[i + 1]

        pc, note = _find_crossing(p_vals, s1, s2)
        if pc is not None:
            unc = _bootstrap_crossing_uncertainty(p_vals, s1, s2, e1, e2,
                                                  rng_seed=i)
        else:
            unc = float('nan')

        rows.append({
            'L1':         L1,
            'L2':         L2,
            'pcross_eff': pc,
            'uncertainty': unc,
            'note':        note,
        })
    return rows


def _format_crossing_table(rows, mean_cross, largest_cross, largest_unc, spread):
    lines = []
    lines.append("=" * 72)
    lines.append("Adjacent-Size Crossing Analysis  (pcross_eff, not thermodynamic p_c)")
    lines.append("Bootstrap uncertainty = statistical precision at fixed L only.")
    lines.append("Spread across pairs = systematic finite-size drift (NOT stat. error).")
    lines.append("=" * 72)
    lines.append(f"{'L1':>5}  {'L2':>5}  {'pcross_eff':>12}  "
                 f"{'boot_unc':>10}  Note")
    lines.append("-" * 72)
    for r in rows:
        pc_str  = f"{r['pcross_eff']:.4f}" if r['pcross_eff'] is not None else "  none "
        un_str  = f"{r['uncertainty']:.4f}" if np.isfinite(r['uncertainty']) else "  n/a  "
        note    = r['note'] or ""
        lines.append(f"{r['L1']:>5}  {r['L2']:>5}  {pc_str:>12}  {un_str:>10}  {note}")
    lines.append("-" * 72)

    mc_str  = f"{mean_cross:.4f}" if mean_cross is not None else "n/a"
    lc_str  = f"{largest_cross:.4f}" if largest_cross is not None else "n/a"
    lu_str  = f"+/- {largest_unc:.4f}" if np.isfinite(largest_unc) else ""
    sp_str  = f"{spread:.4f}" if spread is not None else "n/a"
    lines.append(f"  Mean crossing    (all pairs):   pcross_eff = {mc_str}")
    lines.append(f"  Largest-L crossing ({rows[-1]['L1']} x {rows[-1]['L2']}):"
                 f"  pcross_eff = {lc_str} {lu_str}")
    lines.append(f"  Spread (max - min):             {sp_str}")
    lines.append("  NOTE: spread > boot_unc signals systematic finite-size drift")
    lines.append("=" * 72)
    return "\n".join(lines)


def main():
    (L_vals, p_vals, mean_S, std_S, sem_S), source = _load_best()
    n_L = len(L_vals)

    # ------------------------------------------------------------------ #
    # Crossing analysis
    # ------------------------------------------------------------------ #
    cross_rows = _compute_crossing_table(L_vals, p_vals, mean_S, sem_S)

    valid_crosses = [r['pcross_eff'] for r in cross_rows
                     if r['pcross_eff'] is not None]
    mean_cross    = float(np.mean(valid_crosses)) if valid_crosses else None
    largest_cross = cross_rows[-1]['pcross_eff'] if cross_rows else None
    largest_unc   = cross_rows[-1]['uncertainty'] if cross_rows else float('nan')
    spread        = ((max(valid_crosses) - min(valid_crosses))
                     if len(valid_crosses) >= 2 else None)

    cross_table = _format_crossing_table(
        cross_rows, mean_cross, largest_cross, largest_unc, spread)
    print(cross_table)

    # Save crossing table
    os.makedirs(os.path.dirname(OUT_CROSS_TXT), exist_ok=True)
    with open(OUT_CROSS_TXT, 'w') as f:
        f.write(cross_table + "\n")
        # Add machine-readable mean crossing so log_scaling.py can parse it
        if mean_cross is not None:
            f.write(f"\nmean crossing = {mean_cross:.6f}\n")
    print(f"Saved crossing table: {OUT_CROSS_TXT}")

    # Save as JSON for downstream consumption
    cross_json = {
        'pairs':           cross_rows,
        'mean_pcross_eff': mean_cross,
        'largest_pcross_eff':  largest_cross,
        'largest_pcross_unc':  largest_unc if np.isfinite(largest_unc) else None,
        'spread':          spread,
        'note': ('Spread reflects systematic finite-size drift; '
                 'bootstrap uncertainties are statistical only at fixed L'),
    }
    with open(OUT_CROSS_JSON, 'w') as f:
        json.dump(cross_json, f, indent=2)
    print(f"Saved crossing JSON:  {OUT_CROSS_JSON}")

    # ------------------------------------------------------------------ #
    # Figure
    # ------------------------------------------------------------------ #
    cmap   = matplotlib.colormaps.get_cmap("plasma").resampled(n_L + 2)
    colors = [cmap(n_L + 1 - i) for i in range(n_L)]

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, L in enumerate(L_vals):
        c  = colors[i]
        ls = LINESTYLES[i % len(LINESTYLES)]
        ax.plot(p_vals, mean_S[i], color=c, linewidth=2, linestyle=ls,
                label=f"$L = {L}$", zorder=3)
        ax.fill_between(p_vals,
                        mean_S[i] - sem_S[i],
                        mean_S[i] + sem_S[i],
                        color=c, alpha=0.25, zorder=2)

    # Mark literature p_c and the mean finite-size effective crossing
    ax.axvline(0.16, color="dimgray", linestyle="--", linewidth=1.4,
               label=r"$p_c^{\mathrm{lit}} \approx 0.16$", zorder=4)
    if mean_cross is not None:
        ax.axvline(mean_cross, color="tomato", linestyle=":", linewidth=1.4,
                   label=f"$p_c^{{\\mathrm{{cross,eff}}}} \\approx {mean_cross:.3f}$",
                   zorder=4)

    y_mid = float(np.nanmax(mean_S)) * 0.55
    y_low = float(np.nanmax(mean_S)) * 0.12
    ax.text(0.05, y_mid, r"Volume-law  $S \propto L$",
            fontsize=11, color="#1A5276",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#D6EAF8",
                      edgecolor="#1A5276", alpha=0.8))
    ax.text(0.32, y_low, r"Area-law  $S = \mathcal{O}(1)$",
            fontsize=11, color="#7B241C",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FADBD8",
                      edgecolor="#7B241C", alpha=0.8))

    ax.set_xlabel(r"Measurement rate  $p$")
    ax.set_ylabel(r"Half-chain entropy  $S(L/2)$  [ebits]")
    ax.set_title("Finite-Size Signatures of the MIPT\n"
                 r"(crossing spread = finite-size drift, not statistical error)")
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(bottom=0)
    ax.set_xticks(np.arange(0.0, 0.55, 0.1))
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, which="major", linestyle="--", color="lightgray", alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_PNG}")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n--- pcross_eff summary ---")
    for r in cross_rows:
        pc_str = f"{r['pcross_eff']:.4f}" if r['pcross_eff'] is not None else "none"
        un_str = f"+/- {r['uncertainty']:.4f}" if np.isfinite(r['uncertainty']) else ""
        print(f"  L={r['L1']:>2} x L={r['L2']:>2}:  pcross_eff = {pc_str} {un_str}"
              + (f"  [{r['note']}]" if r['note'] else ""))
    if mean_cross is not None:
        print(f"\n  Mean crossing    (all pairs):   {mean_cross:.4f}")
    if largest_cross is not None:
        luc = f"+/- {largest_unc:.4f}" if np.isfinite(largest_unc) else ""
        print(f"  Largest-L crossing:             {largest_cross:.4f} {luc}")
    if spread is not None:
        print(f"  Spread (max - min):             {spread:.4f}  "
              "(systematic finite-size drift)")
    print("\n  Literature p_c = 0.16  (Li 2019; L ~ 512)")
    print("  Upward drift of pcross_eff is expected and well-documented "
          "for L <= 50.")


if __name__ == "__main__":
    main()
