# Measurement-Induced Entanglement Transition (MIET)

Numerical study of the entanglement crossover in random Clifford circuits with
projective measurements, using the stabilizer tableau formalism.

See the [project README](../README.md) for full documentation, results summary,
and scientific limitations.

## Quick start

Run from the **repo root** (`miet-clifford/`):

```bash
pip install -e ".[dev]"
pytest -v
python miet/scripts/physics_audit.py
```

## Full sweep and figures

Scripts that save to `data/` must run from `miet/` (save path is CWD-relative).
Analysis scripts run from the repo root.

```bash
# From miet/:
cd miet
python scripts/run_quick.py --seed 42    # ~1 min, L in {8,12}
# or full 2-3 hour sweep:
# python scripts/run_sweep.py --seed 42

# From repo root:
cd ..
python miet/analysis/phase_diagram.py
python miet/analysis/finite_size.py
python miet/analysis/log_scaling.py
python miet/analysis/critical_fit.py
```
