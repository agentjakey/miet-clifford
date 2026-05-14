# miet-clifford

Numerical study of measurement-induced entanglement transitions (MIETs) in
one-dimensional random Clifford circuits, implemented from scratch using the
Aaronson-Gottesman stabilizer tableau formalism.

PHYS 130B final project — University of California San Diego.

> **Note:** This project reproduces qualitative finite-size signatures of the
> measurement-induced entanglement transition. The reported values of
> $p_c$, $\nu$, and $\alpha$ are finite-size effective estimates obtained from
> system sizes $L \leq 24$, not precision thermodynamic estimates. See
> [Scientific limitations](#scientific-limitations) below.

---

## What this project does

A hybrid quantum circuit alternates brickwork layers of random two-qubit
Clifford gates with single-site projective measurements applied at rate $p$.
As $p$ increases, the system crosses from a volume-law entangled phase
($S(L/2) \propto L$) to an area-law phase ($S(L/2) = O(1)$).

This code:
- Implements the full Aaronson-Gottesman $(2n) \times (2n+1)$ binary tableau
  over GF(2) from scratch, without Stim or Qiskit.
- Computes entanglement entropy via GF(2) rank (XOR Gaussian elimination),
  not floating-point linear algebra.
- Sweeps $(L, p)$ with disorder averaging and checkpoint-saves results.
- Generates four figures: circuit schematic, phase diagram, log-scaling fit,
  and a finite-size scaling collapse.
- Runs a seven-check physics audit to verify correctness of the simulation.
- Produces a compiled LaTeX report in RevTeX4-2 / PRL format.

## Results summary

| Quantity | Finite-size eff. ($L \leq 24$) | Literature (thermodynamic) |
|---|---|---|
| $p_c$ (curve crossing) | $0.203 \pm 0.017$ | $\approx 0.16$ (Li et al. 2019) |
| $p_c$ (FSS collapse) | $0.209 \pm 0.008$ | $\approx 0.16$ (Li et al. 2019) |
| $\nu$ | $1.72 \pm 0.10$ | $\approx 1.28$ (Zabalo et al. 2020) |
| $\alpha$ at $p = 0.16$ | $0.27 \pm 0.07$ | $\approx 1.6$ (Li et al. 2019, $L \gg 100$) |
| $\alpha_{\mathrm{vol}}$ at $p = 0$ | $0.995 \pm 0.017$ | $\sim 1$ (volume-law, system-dependent) |

Reported $\pm$ values are bootstrap statistical uncertainties only.
Systematic finite-size bias is the dominant source of deviation from
thermodynamic values and cannot be reduced by collecting more samples at
fixed $L$.

## Scientific limitations

- **System sizes:** $L \in \{8, 12, 16, 20, 24\}$. These sizes are sufficient
  to show qualitative finite-size signatures of the MIPT crossover but are too
  small for reliable extraction of thermodynamic critical exponents. Precision
  MIPT studies use $L$ up to $\sim 512$.

- **Disorder averaging:** 200 independent realizations per $(L, p)$ point.
  Statistical uncertainties are small relative to systematic finite-size bias.

- **Finite-size corrections:** The effective critical rate
  $p_c^{\mathrm{eff}} \approx 0.20$ is above the thermodynamic value
  $p_c \approx 0.16$ due to finite-size corrections, not a physical discrepancy.
  This upward bias is well-documented in the MIPT literature at $L \lesssim 50$.

- **FSS collapse:** The collapse uses the simplified ansatz
  $S(L,p) = f((p - p_c) \cdot L^{1/\nu})$, which omits the additive logarithmic
  term present at criticality. The extracted $p_c^{\mathrm{eff}}$ and
  $\nu_{\mathrm{eff}}$ are diagnostic, not precision thermodynamic estimates.

- **Log-scaling fit:** The coefficient $\alpha_{\mathrm{eff}} = 0.27$ is fit
  from five system sizes evaluated at $p = 0.16$, which lies below the
  finite-size effective critical rate. The fit is underpowered and sensitive
  to the chosen evaluation point. It is not quantitatively consistent with
  the asymptotic value $\alpha \approx 1.6$ (established at $L \gg 100$).

- **Gate ensemble:** Random two-qubit Cliffords are drawn from the decomposition
  $(1\mathrm{Q}) \cdot \mathrm{CNOT}_{01} \cdot (1\mathrm{Q}) \cdot
  \mathrm{CNOT}_{10} \cdot (1\mathrm{Q}) \cdot (1\mathrm{Q})$ with each 1Q
  drawn uniformly from the 24-element single-qubit Clifford group. This ensemble
  is expected to produce scrambling Clifford dynamics in the MIPT universality
  class and matches the construction used in the literature. It is not verified
  to be a uniform sampler over the full 11,520-element two-qubit Clifford group.
  **Future work:** Implement or verify exact uniform two-qubit Clifford sampling.

## Reproducing results

All commands run from the `miet/` directory.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the test suite

```bash
pytest tests/ -v
```

The tests cover: tableau initialization, Bell state entropy (primary correctness
check), GHZ state, measurement collapse, Bell correlations, H-gate randomness,
S-gate phase, GF(2) rank, deterministic measurement (Cases 1 and 2), and
circuit-level volume-law / area-law behavior.

### 3. Run the physics audit (7 correctness checks)

```bash
python scripts/physics_audit.py
```

Output is written to `data/physics_audit.txt`. All 7 checks must pass before
trusting any numerical results.

### 4. Quick simulation (small sizes, fast)

```bash
python scripts/run_quick.py
```

Runs a reduced $(L, p)$ sweep on $L \in \{4, 6, 8\}$ for fast sanity checking.
Results are saved to `data/quick_results.npz`.

### 5. Full parameter sweep

```bash
python scripts/run_sweep.py
```

Sweeps $L \in \{8, 12, 16, 20, 24\}$ over 26 values of $p \in [0, 0.5]$ with
200 disorder realizations per point. Results are checkpoint-saved to
`data/sweep_results.npz`. Wall-clock time: approximately 2-3 hours on a single
CPU core.

### 6. Regenerate figures

Run after the full sweep completes:

```bash
python analysis/phase_diagram.py   # fig1_phase_diagram
python analysis/log_scaling.py     # fig2_log_scaling
python analysis/finite_size.py     # fig3_scaling_collapse
python analysis/critical_fit.py    # prints critical quantities to data/
```

Figures are written to `figures/` as both PDF and PNG.

### 7. Rebuild the report (requires MiKTeX or TeX Live)

```bash
cd report/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The compiled PDF is also provided at the repo root as `miet_research_report.pdf`.

## Repository structure

```
miet-clifford/
  miet/
    src/
      stabilizer.py     # Aaronson-Gottesman tableau (from scratch)
      entropy.py        # GF(2) rank entanglement entropy
      circuit.py        # brickwork hybrid circuit
      simulation.py     # disorder averaging, parameter sweep, checkpointing
    analysis/
      phase_diagram.py
      log_scaling.py
      finite_size.py
      critical_fit.py
    scripts/
      run_quick.py
      run_sweep.py
      physics_audit.py
    tests/
      test_stabilizer.py
    data/               # simulation outputs (npz, txt)
    figures/            # generated figures (pdf, png)
    report/             # LaTeX source (RevTeX4-2 / PRL format)
    requirements.txt
  miet_research_report.pdf   # compiled report
```

## References

- Li, Chen, Fisher. PRB 98, 205136 (2018); PRB 100, 134306 (2019).
- Skinner, Ruhman, Nahum. PRX 9, 031009 (2019).
- Aaronson, Gottesman. PRA 70, 052328 (2004).
- Hamma, Ionicioiu, Zanardi. PLA 337, 22 (2005).
- Jian, You, Vasseur, Ludwig. PRB 101, 104302 (2020).
- Zabalo et al. PRB 101, 060301(R) (2020).
