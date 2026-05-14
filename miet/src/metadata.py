"""
Metadata collection and persistence for simulation runs.

Every .npz results file gets a companion _metadata.json written next to it,
capturing the full provenance needed to reproduce or interpret the data.
"""
import json
import sys
import datetime as _dt
import numpy as np


def _git_hash():
    try:
        import subprocess
        r = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def collect_metadata(L_values, p_values, n_samples,
                     n_steps_fn, warmup_fn, seed,
                     measurements_during_warmup=True):
    """Build the metadata dict for one parameter sweep."""
    L_list = [int(L) for L in L_values]
    ex = L_list[0] if L_list else 0
    return {
        'L_values':                           L_list,
        'p_values':                           [float(p) for p in p_values],
        'n_samples':                          n_samples,
        'warmup_rule':                        f'warmup_fn(L); L={ex} -> {warmup_fn(ex)}',
        'main_layers_rule':                   f'n_steps_fn(L); L={ex} -> {n_steps_fn(ex)}',
        'total_circuit_depth_rule':           'warmup_layers + main_layers per realization',
        'measurements_applied_during_warmup': measurements_during_warmup,
        'gate_ensemble': (
            '6 independent uniform draws from the 24-element single-qubit Clifford group '
            '+ 2 CNOT gates per 2-qubit gate; '
            'not verified as a uniform sampler over the full '
            '11520-element 2-qubit Clifford group'
        ),
        'random_seed':    seed,
        'datetime_utc':   _dt.datetime.now(_dt.timezone.utc).isoformat(),
        'python_version': sys.version,
        'numpy_version':  np.__version__,
        'git_commit_hash': _git_hash(),
        'code_version':   None,
    }


def save_metadata(meta, npz_path):
    """Write meta dict as JSON next to npz_path. Returns the JSON path."""
    base = npz_path[:-4] if npz_path.endswith('.npz') else npz_path
    out = base + '_metadata.json'
    with open(out, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    return out
