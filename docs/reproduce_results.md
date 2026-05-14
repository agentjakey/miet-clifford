# Reproducing Results

All commands run from the **repo root** (`miet-clifford/`) unless noted.

---

## 0. Quickest route: Makefile

A `Makefile` at the repo root encodes the correct workflow order:

```bash
make install    # pip install -e ".[dev]"
make test       # pytest (53 tests)
make audit      # 7 physics correctness checks
make quick      # quick simulation sweep, L in {8,12}
make figures    # regenerate all figures from existing data
make report     # compile LaTeX report (requires pdflatex)
```

Run `make help` for the full target list. The sections below document the same
steps manually for reference.

On Windows with MiKTeX, set the LaTeX path for the `report` target:

```bash
make report PDFLATEX="C:/Users/.../MiKTeX/miktex/bin/x64/pdflatex.exe" \
            BIBTEX="C:/Users/.../MiKTeX/miktex/bin/x64/bibtex.exe"
```

---

## 1. Environment setup

Python 3.9 or later. No compiled extensions required; the stabilizer tableau
is pure Python/NumPy. The simulation has been tested on Linux and Windows.

---

## 2. Dependency installation

From the repo root:

```bash
pip install -e ".[dev]"
```

This installs the `miet_clifford` package in editable mode plus all runtime
dependencies (`numpy`, `scipy`, `matplotlib`, `tqdm`) and `pytest`. The
`pyproject.toml` at the repo root is the authoritative dependency specification.

---

## 3. Running tests

From the repo root:

```bash
pytest -v
```

53 tests cover tableau initialization, gate conjugation (H, S, CNOT), Bell and
GHZ entanglement entropy, measurement collapse and repeatability, GF(2) rank
correctness, symplectic pairing invariants, circuit-level volume-law /
area-law behavior, and Clifford sampler correctness. All 53 must pass before
trusting any numerical results.

---

## 4. Running the physics audit

```bash
# Makefile shortcut:
make audit

# Or directly (from repo root):
python miet/scripts/physics_audit.py
```

Runs 7 correctness checks (Bell entropy = 1 ebit, area-law saturation,
volume-law scaling, measurement collapse, etc.). Output is written to
`miet/data/physics_audit.txt`. All 7 checks must pass. The audit uses a
fixed seed (`AUDIT_SEED = 12345`) and produces identical output on every run.

---

## 5. Running a quick simulation

```bash
# Makefile shortcut:
make quick

# Or directly (must run from miet/ because save path is CWD-relative):
cd miet && python scripts/run_quick.py --seed 42
```

Sweeps `L in {8, 12}` over 11 values of `p` in `[0, 0.5]` with 30 disorder
realizations per point. Completes in under 60 seconds on a modern CPU.
Results are checkpoint-saved to `miet/data/quick_results.npz`.

The analysis scripts fall back to `quick_results.npz` when `sweep_results.npz`
is absent, so figures can be previewed immediately after this step (with
reduced quality).

---

## 6. Running the full simulation sweep

```bash
# From miet/ (save path is CWD-relative):
cd miet && python scripts/run_sweep.py --seed 42
```

Sweeps `L in {8, 12, 16, 20, 24}` over 26 values of `p` in `[0, 0.5]` with
200 disorder realizations per `(L, p)` point (26,000 total circuit runs).
Results are checkpoint-saved to `miet/data/sweep_results.npz` and a provenance
record is written to `miet/data/sweep_results_metadata.json`.

This target is intentionally not in the Makefile to prevent accidental 2-3
hour runs.

---

## 7. Regenerating figures

```bash
# Makefile shortcut (runs scripts in correct dependency order):
make figures

# Or individually from repo root:
python miet/analysis/circuit_schematic.py   # fig0 -- no data dependency
python miet/analysis/phase_diagram.py       # fig1 -- writes crossing_table.txt/json
python miet/analysis/finite_size.py         # fig3 -- writes fss_config.json
python miet/analysis/log_scaling.py         # fig2 -- reads crossing_table.txt, fss_config.json
python miet/analysis/critical_fit.py        # tabular summary -- writes critical_quantities.txt
```

All figures are written to `miet/figures/` as both PDF and PNG.

---

## 8. Rebuilding the report

```bash
# Makefile shortcut:
make report

# On Windows with MiKTeX:
make report PDFLATEX="C:/Users/.../MiKTeX/miktex/bin/x64/pdflatex.exe" \
            BIBTEX="C:/Users/.../MiKTeX/miktex/bin/x64/bibtex.exe"

# Or manually from miet/report/:
cd miet/report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Requires MiKTeX or TeX Live with RevTeX4-2.

The compiled PDF is also provided at the repo root as `miet_research_report.pdf`.
Copy it there after rebuilding:

```bash
cp report/main.pdf ../miet_research_report.pdf
```

---

## 9. Expected runtime

| Step | Approximate time |
|---|---|
| `run_quick.py` | < 1 minute |
| `run_sweep.py` | 2-3 hours (single CPU core) |
| All analysis scripts | < 2 minutes total |
| LaTeX compile (4 passes) | < 30 seconds |

The sweep runtime scales roughly as `sum_L (L * n_p * n_samples)`. On a
multi-core machine, parallelism is not currently implemented; the inner loop
over disorder realizations is sequential.

---

## 10. Expected output files

After a complete run, the following files should exist under `miet/`:

```
data/
  sweep_results.npz              # main simulation data (L, p, mean_S, std_S, sem_S)
  sweep_results_metadata.json    # generated alongside sweep_results.npz during run_sweep.py;
                                 # not committed (records datetime, git hash, seed at run time)
  quick_results.npz              # small sanity-check sweep
  quick_results_metadata.json    # generated alongside quick_results.npz during run_quick.py;
                                 # not committed (same provenance fields as above)
  physics_audit.txt              # 7 correctness checks
  crossing_table.txt             # adjacent-size pcross_eff per pair + summary
  crossing_table.json            # same data in machine-readable form
  fss_config.json                # pc_eff, nu_eff from FSS collapse (read by log_scaling)
  fss_sensitivity.txt            # sensitivity of FSS fit to L range and p range
  alpha_vs_p.txt                 # alpha_eff table over p in [0.10, 0.30]
  critical_quantities.txt        # tabular summary of all extracted quantities

figures/
  fig0_circuit_schematic.pdf/png
  fig1_phase_diagram.pdf/png
  fig2_log_scaling.pdf/png
  fig2b_alpha_vs_p.pdf/png       # supplemental: alpha_eff vs p
  fig3_scaling_collapse.pdf/png
```

---

## 11. Random seed reproducibility

All stochastic operations thread through `numpy.random.Generator` instances
created via `numpy.random.SeedSequence`. Passing `--seed N` to any script
produces bitwise-identical results across runs on the same platform and NumPy
version. Without `--seed`, each run draws from a fresh system-entropy seed
(non-deterministic).

The seed hierarchy is:

```
SeedSequence(master_seed)
  -> spawn one child per (L, p) point
       -> disorder_average() creates one Generator per sample
            -> run_circuit() uses that Generator for all gates and measurements
```

This means individual `(L, p)` point results are independent and their RNG
streams do not interact, even if the sweep is run in a different order. A
metadata JSON recording the master seed is written alongside every `.npz` file.

---

## 12. Known limitations

- **System sizes:** `L in {8, 12, 16, 20, 24}`. These are sufficient to show
  qualitative MIPT phenomenology but too small for reliable thermodynamic
  exponent extraction. Precision studies use L up to ~512.

- **Effective critical rate:** `p_c_eff ~ 0.20` is above the thermodynamic
  value `p_c ~ 0.16`. This upward bias is well-documented in the literature
  at `L <= 50` and is a finite-size effect, not a physical discrepancy.

- **FSS collapse:** The simplified ansatz `S(L,p) = f((p - p_c) * L^{1/nu})`
  omits the additive logarithmic term at criticality. Extracted `pc_eff` and
  `nu_eff` are diagnostic estimates, not precision thermodynamic values.

- **Log-scaling coefficient:** `alpha_eff = 0.27` is fit from five system
  sizes at `p = 0.16`, which lies below the finite-size effective critical
  rate. It is not quantitatively consistent with the asymptotic literature
  value `alpha ~ 1.6` established at `L >> 100`.

- **Bootstrap uncertainties** are statistical precision at fixed L only. The
  dominant source of deviation from literature values is systematic finite-size
  bias, which is not reduced by collecting more disorder realizations.

- **Gate ensemble:** Each single-qubit gate is sampled uniformly from the
  verified 24-element single-qubit Clifford group (the list is programmatically
  enumerated by Pauli action and confirmed to have 24 distinct elements at
  import time). Two-qubit gates use the decomposition
  `(1Q)·CNOT_01·(1Q)·CNOT_10·(1Q)·(1Q)` with six independent uniform 1Q
  draws and two CNOTs. The full two-qubit ensemble is not verified as a
  uniform sampler over the 11,520-element two-qubit Clifford group.

---

## 13. Figure-to-data mapping

### Figure 1 -- Circuit schematic

| Field | Value |
|---|---|
| Script | `analysis/circuit_schematic.py` |
| Data file | None (schematic only, no simulation data) |
| Output | `figures/fig0_circuit_schematic.pdf/png` |
| Report | Fig. 1 in `report/main.tex` |

### Figure 2 -- Phase diagram (S(L/2) vs p)

| Field | Value |
|---|---|
| Script | `analysis/phase_diagram.py`, function `main()` |
| Data file | `data/sweep_results.npz` (falls back to `quick_results.npz`) |
| Output | `figures/fig1_phase_diagram.pdf/png` |
| Side outputs | `data/crossing_table.txt`, `data/crossing_table.json` |
| Report | Fig. 2 in `report/main.tex` |

Key intermediate: `crossing_table.txt` contains the per-pair `pcross_eff`
values and a machine-readable `mean crossing = X` line consumed by
`log_scaling.py`.

### Figure 3 -- Log-scaling fit (S vs ln(L/2))

| Field | Value |
|---|---|
| Script | `analysis/log_scaling.py`, function `main()` |
| Data file | `data/sweep_results.npz` |
| Auxiliary inputs | `data/fss_config.json`, `data/crossing_table.txt` |
| Output | `figures/fig2_log_scaling.pdf/png` |
| Side outputs | `figures/fig2b_alpha_vs_p.pdf/png`, `data/alpha_vs_p.txt` |
| Report | Fig. 3 in `report/main.tex` |

### Figure 4 -- FSS collapse

| Field | Value |
|---|---|
| Script | `analysis/finite_size.py`, function `main()` |
| Data file | `data/sweep_results.npz` |
| Output | `figures/fig3_scaling_collapse.pdf/png` |
| Side outputs | `data/fss_config.json`, `data/fss_sensitivity.txt` |
| Report | Fig. 4 in `report/main.tex` |

`fss_config.json` stores the optimized `pc_eff` and `nu_eff` and must be
generated before running `log_scaling.py`.
