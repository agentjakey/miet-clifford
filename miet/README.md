# Measurement-Induced Entanglement Transition (MIET)

Numerical study of the entanglement phase transition in random Clifford circuits with projective measurements, using the stabilizer tableau formalism.

## Install

```bash
pip install -r requirements.txt
```

## Usage

Run a quick sanity check on small system sizes:

```bash
python scripts/run_quick.py
```

Run the full (L, p) parameter sweep (writes results to `data/`):

```bash
python scripts/run_sweep.py
```

Generate figures from saved data (writes to `figures/`):

```bash
python analysis/phase_diagram.py
python analysis/log_scaling.py
python analysis/finite_size.py
python analysis/critical_fit.py
```
