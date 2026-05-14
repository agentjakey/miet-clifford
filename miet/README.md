# Measurement-Induced Entanglement Transition (MIET)

Numerical study of the entanglement crossover in random Clifford circuits with
projective measurements, using the stabilizer tableau formalism.

See the [project README](../README.md) for full documentation, results summary,
and scientific limitations.

## Quick start

```bash
pip install -r requirements.txt
pytest tests/ -v
python scripts/physics_audit.py
python scripts/run_quick.py
```

## Full sweep and figures

```bash
python scripts/run_sweep.py
python analysis/phase_diagram.py
python analysis/log_scaling.py
python analysis/finite_size.py
python analysis/critical_fit.py
```
