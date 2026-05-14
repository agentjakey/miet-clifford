"""
Critical quantities summary for the MIPT.

Computes and tabulates p_cross_eff, p_fss_eff, nu_eff, alpha_eff, alpha_vol,
and S_inf by re-running the estimators from Figs 1-3 against the best
available data.

All extracted quantities are finite-size effective estimates from L <= 24.
They are not thermodynamic values: the dominant source of deviation from
literature is systematic finite-size bias, not statistical noise.

NOTE on alpha_eff:
  alpha_eff is the coefficient in S(L/2) = alpha_eff*ln(L/2) + const at p_c.
  Li et al. (2019) PRB 100, 134306 report alpha(p_c) ~ 1.6 for random
  Clifford circuits at L >> 100.  We do NOT convert to c_eff = 6*alpha_eff:
  the MIPT is not a standard CFT (Jian et al. PRB 101, 104302, 2020).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import linregress
from scipy.interpolate import interp1d
import os, sys, warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.simulation import load_results

SWEEP_PATH = os.path.join(os.path.dirname(__file__), "../data/sweep_results.npz")
QUICK_PATH = os.path.join(os.path.dirname(__file__), "../data/quick_results.npz")
OUT_TXT    = os.path.join(os.path.dirname(__file__), "../data/critical_quantities.txt")


# ── Data loading ──────────────────────────────────────────────────────────────

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


# ── Fig 1: curve crossing ─────────────────────────────────────────────────────

def _find_crossing(p, s1, s2):
    """Linear interpolation of crossing between two S(p) curves."""
    diff = s1 - s2
    changes = np.where(np.diff(np.sign(diff)))[0]
    if len(changes) == 0:
        return None
    i = changes[0]
    d0, d1 = diff[i], diff[i + 1]
    return float(p[i] - d0 * (p[i + 1] - p[i]) / (d1 - d0))


def _pc_from_crossing(L_vals, p_vals, mean_S, sem_S, n_boot=200):
    """
    Cross adjacent L pairs, average crossings, bootstrap uncertainty.
    Uses the two largest L values to get the best finite-size estimate.
    """
    if len(L_vals) < 2:
        return float('nan'), float('nan')

    # Primary estimate: crossing of the two largest L curves
    crossings = []
    for i in range(len(L_vals) - 1):
        x = _find_crossing(p_vals, mean_S[i], mean_S[i + 1])
        if x is not None:
            crossings.append(x)
    pc_est = float(np.mean(crossings)) if crossings else float('nan')

    # Bootstrap
    boot_vals = []
    for _ in range(n_boot):
        noise = np.random.randn(*mean_S.shape) * sem_S
        S_b = mean_S + noise
        xs = []
        for i in range(len(L_vals) - 1):
            x = _find_crossing(p_vals, S_b[i], S_b[i + 1])
            if x is not None:
                xs.append(x)
        if xs:
            boot_vals.append(float(np.mean(xs)))
    pc_err = float(np.std(boot_vals, ddof=1)) if len(boot_vals) > 1 else float('nan')
    return pc_est, pc_err


# ── Fig 3: FSS collapse ───────────────────────────────────────────────────────

def _collapse_cost(params, L_vals, p_vals, mean_S):
    p_c, nu = params
    if nu <= 0:
        return 1e9
    curves = []
    for i, L in enumerate(L_vals):
        x = (p_vals - p_c) * (L ** (1.0 / nu))
        order = np.argsort(x)
        curves.append((x[order], mean_S[i][order]))

    x_lo = max(c[0][0]  for c in curves)
    x_hi = min(c[0][-1] for c in curves)
    if x_lo >= x_hi:
        return 1e9

    x_grid = np.linspace(x_lo, x_hi, 60)
    interps = []
    for x_c, y_c in curves:
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


def _fss_optimize(L_vals, p_vals, mean_S, p_c0=0.16, nu0=1.3):
    bounds_pc = (0.08, 0.28)
    bounds_nu = (0.5, 3.0)

    def cost_b(params):
        p_c, nu = params
        if not (bounds_pc[0] <= p_c <= bounds_pc[1]):
            return 1e9
        if not (bounds_nu[0] <= nu <= bounds_nu[1]):
            return 1e9
        return _collapse_cost(params, L_vals, p_vals, mean_S)

    res = minimize(cost_b, x0=[p_c0, nu0], method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 4000})
    return float(res.x[0]), float(res.x[1])


def _fss_fit(L_vals, p_vals, mean_S, sem_S, n_boot=100):
    pc_opt, nu_opt = _fss_optimize(L_vals, p_vals, mean_S)
    pc_boot, nu_boot = [], []
    for _ in range(n_boot):
        noise = np.random.randn(*mean_S.shape) * sem_S
        pc_b, nu_b = _fss_optimize(L_vals, p_vals, mean_S + noise,
                                   p_c0=pc_opt, nu0=nu_opt)
        pc_boot.append(pc_b)
        nu_boot.append(nu_b)
    pc_err = float(np.std(pc_boot, ddof=1)) if len(pc_boot) > 1 else float('nan')
    nu_err = float(np.std(nu_boot, ddof=1)) if len(nu_boot) > 1 else float('nan')
    return pc_opt, pc_err, nu_opt, nu_err


# ── Fig 2: log scaling alpha ──────────────────────────────────────────────────

def _alpha_fit(L_vals, p_vals, mean_S, sem_S, n_boot=200):
    """Fit S(L/2) = alpha*ln(L/2) + const at the p closest to 0.16."""
    p_idx = int(np.argmin(np.abs(p_vals - 0.16)))
    x = np.array([np.log(L / 2.0) for L in L_vals])
    y = mean_S[:, p_idx]
    e = sem_S[:, p_idx]

    if len(x) < 2:
        return float('nan'), float('nan'), float(p_vals[p_idx])

    fit = linregress(x, y)
    alpha = float(fit.slope)

    # Bootstrap
    alpha_boot = []
    for _ in range(n_boot):
        y_b = y + np.random.randn(len(y)) * e
        alpha_boot.append(float(linregress(x, y_b).slope))
    alpha_err = float(np.std(alpha_boot, ddof=1))
    return alpha, alpha_err, float(p_vals[p_idx])


# ── Volume-law coefficient ────────────────────────────────────────────────────

def _alpha_vol_fit(L_vals, p_vals, mean_S, sem_S, n_boot=200):
    """Fit S(L/2) = alpha_vol*(L/2) + const at p=0 using largest L values."""
    p_idx = int(np.argmin(np.abs(p_vals - 0.0)))

    # Use up to the 3 largest L values
    n_use = min(3, len(L_vals))
    idx   = slice(-n_use, None)
    L_use = L_vals[idx] if not isinstance(L_vals, list) else L_vals[-n_use:]
    x = np.array([L / 2.0 for L in L_use])
    y = mean_S[idx if not isinstance(idx, slice) else -n_use:, p_idx]
    e = sem_S[-n_use:, p_idx]

    if len(x) < 2:
        return float('nan'), float('nan')

    fit = linregress(x, y)
    alpha_vol = float(fit.slope)

    boot = []
    for _ in range(n_boot):
        y_b = y + np.random.randn(len(y)) * e
        boot.append(float(linregress(x, y_b).slope))
    alpha_vol_err = float(np.std(boot, ddof=1))
    return alpha_vol, alpha_vol_err


# ── Area-law saturation ───────────────────────────────────────────────────────

def _area_law(L_vals, p_vals, mean_S, sem_S):
    """Mean and SEM of S at p=0.5 for the largest L."""
    p_idx = int(np.argmin(np.abs(p_vals - 0.5)))
    s_val = float(mean_S[-1, p_idx])
    s_err = float(sem_S[-1, p_idx])
    return s_val, s_err


# ── Table formatting ──────────────────────────────────────────────────────────

def _fmt(val, err, width=17):
    if np.isfinite(val) and np.isfinite(err):
        s = f"{val:.3f} +/- {err:.3f}"
    elif np.isfinite(val):
        s = f"{val:.3f}"
    else:
        s = "N/A"
    return s.ljust(width)


def _build_table(rows):
    col_q   = max(len(r[0]) for r in rows) + 2
    col_tw  = 19
    col_lit = max(len(r[2]) for r in rows) + 2
    sep = (f"+{'-'*(col_q)}+{'-'*(col_tw)}+{'-'*(col_lit)}+")
    hdr = (f"| {'Quantity':<{col_q-2}} | {'This work':<{col_tw-2}} "
           f"| {'Literature':<{col_lit-2}} |")
    lines = [sep, hdr, sep]
    for q, val_s, lit in rows:
        lines.append(
            f"| {q:<{col_q-2}} | {val_s:<{col_tw-2}} | {lit:<{col_lit-2}} |"
        )
    lines.append(sep)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    L_vals, p_vals, mean_S, std_S, sem_S = _load_best()
    n_L = len(L_vals)
    n_boot = 200

    print(f"\nComputing critical quantities for L = {L_vals} ...")

    print("  [1/5] p_c from curve crossing ...")
    pc_cross, pc_cross_err = _pc_from_crossing(
        L_vals, p_vals, mean_S, sem_S, n_boot=n_boot)

    print("  [2/5] FSS collapse (p_c, nu) ...")
    pc_fss, pc_fss_err, nu, nu_err = _fss_fit(
        L_vals, p_vals, mean_S, sem_S, n_boot=100)

    print("  [3/5] Log-scaling alpha at p_c ...")
    alpha, alpha_err, p_used = _alpha_fit(
        L_vals, p_vals, mean_S, sem_S, n_boot=n_boot)

    print("  [4/5] Volume-law coefficient alpha_vol at p=0 ...")
    alpha_vol, alpha_vol_err = _alpha_vol_fit(
        L_vals, p_vals, mean_S, sem_S, n_boot=n_boot)

    print("  [5/5] Area-law saturation at p=0.5 ...")
    s_inf, s_inf_err = _area_law(L_vals, p_vals, mean_S, sem_S)

    rows = [
        ("p_cross_eff (crossing of S curves, Fig 1)",
         _fmt(pc_cross, pc_cross_err),
         "~0.16  (Li+ 2019)"),
        ("p_fss_eff (FSS collapse, Fig 3)",
         _fmt(pc_fss, pc_fss_err),
         "~0.16  (Li+ 2019)"),
        ("nu_eff (correlation exponent, Fig 3)",
         _fmt(nu, nu_err),
         "~1.3   (Zabalo+ 2020)"),
        (f"alpha_eff (log-scaling at p={p_used:.3f}, Fig 2)",
         _fmt(alpha, alpha_err),
         "~1.6   (Li+ 2019)"),
        ("alpha_vol (volume-law coeff, p=0)",
         _fmt(alpha_vol, alpha_vol_err),
         "~1     (system dependent)"),
        (f"S_inf (area-law saturation, L={L_vals[-1]}, p=0.5)",
         _fmt(s_inf, s_inf_err),
         "O(1)   (area law)"),
    ]

    table = _build_table(rows)
    note = (
        "\nNOTE: alpha_eff is the coefficient in S(L/2) = alpha_eff * ln(L/2) + const at p_c.\n"
        "Li et al. (2019) PRB 100, 134306 report alpha(p_c) ~ 1.6 for random Clifford circuits at L >> 100.\n"
        "Do NOT convert to c_eff = 6*alpha_eff: the MIPT is not a standard CFT.\n"
        "(Jian, You, Vasseur, Ludwig, PRB 101, 104302, 2020)\n"
    )
    if n_L < 3:
        note += (f"\nCAVEAT: only {n_L} system size(s) available. "
                 "FSS and log-scaling fits are under-constrained.\n"
                 "Run scripts/run_sweep.py to completion for reliable exponents.\n")

    header = (
        "MIPT Critical Quantities\n"
        + "=" * 70 + "\n\n"
        + "NOTE: All p_c, nu, and alpha quantities reported here are finite-size\n"
        + "effective estimates from L <= 24 unless explicitly labeled as literature\n"
        + "values. Systematic finite-size bias dominates over statistical uncertainty.\n\n"
    )
    output = table + "\n" + note
    print("\n" + output)

    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, "w") as f:
        f.write(header)
        f.write(output)
    print(f"Saved {OUT_TXT}")


if __name__ == "__main__":
    main()
