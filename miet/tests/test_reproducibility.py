"""
Tests for reproducible seeding and metadata persistence.
"""
import json
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.simulation import disorder_average, parameter_sweep


REQUIRED_METADATA_FIELDS = [
    'L_values',
    'p_values',
    'n_samples',
    'warmup_rule',
    'main_layers_rule',
    'total_circuit_depth_rule',
    'measurements_applied_during_warmup',
    'gate_ensemble',
    'random_seed',
    'datetime_utc',
    'python_version',
    'numpy_version',
    'git_commit_hash',
]


# ---------------------------------------------------------------------------
# Reproducibility: disorder_average
# ---------------------------------------------------------------------------

def test_same_seed_identical_samples():
    """Same seed must produce byte-identical sample arrays."""
    r1 = disorder_average(4, 0.2, 8, n_samples=5, seed=42)
    r2 = disorder_average(4, 0.2, 8, n_samples=5, seed=42)
    np.testing.assert_array_equal(r1['samples'], r2['samples'],
        err_msg="disorder_average: same seed produced different samples")


def test_different_seeds_different_samples():
    """Different seeds must produce different sample arrays."""
    r1 = disorder_average(4, 0.2, 8, n_samples=5, seed=42)
    r2 = disorder_average(4, 0.2, 8, n_samples=5, seed=99)
    assert not np.array_equal(r1['samples'], r2['samples']), \
        "disorder_average: different seeds produced identical samples"


def test_samples_are_independent():
    """Different sample indices within one call must not be identical."""
    r = disorder_average(4, 0.2, 16, n_samples=10, seed=7)
    # All 10 values identical would indicate correlated RNG state
    assert len(set(r['samples'].tolist())) > 1, \
        "All samples are identical -- child RNGs are not independent"


# ---------------------------------------------------------------------------
# Reproducibility: parameter_sweep
# ---------------------------------------------------------------------------

def test_parameter_sweep_reproducible(tmp_path):
    """Two sweep runs with the same seed must produce identical mean_entropy arrays."""
    p = str(tmp_path / 'r1.npz')
    q = str(tmp_path / 'r2.npz')
    L_vals = [4]
    p_vals = np.linspace(0.0, 0.5, 5)

    parameter_sweep(L_vals, p_vals, n_samples=4,
                    save_path=p, verbose=False, seed=123)
    parameter_sweep(L_vals, p_vals, n_samples=4,
                    save_path=q, verbose=False, seed=123)

    d1 = np.load(p)
    d2 = np.load(q)
    np.testing.assert_array_equal(d1['mean_entropy'], d2['mean_entropy'],
        err_msg="parameter_sweep: same seed produced different mean_entropy")


# ---------------------------------------------------------------------------
# Metadata: presence and required fields
# ---------------------------------------------------------------------------

def test_metadata_file_created(tmp_path):
    """parameter_sweep must write a _metadata.json alongside the .npz."""
    npz = str(tmp_path / 'out.npz')
    parameter_sweep([4], np.linspace(0.0, 0.5, 3), n_samples=2,
                    save_path=npz, verbose=False, seed=0)

    meta_path = npz.replace('.npz', '_metadata.json')
    assert os.path.exists(meta_path), \
        f"Metadata file not found: {meta_path}"


def test_metadata_contains_required_fields(tmp_path):
    """Metadata JSON must contain all required provenance fields."""
    npz = str(tmp_path / 'out.npz')
    parameter_sweep([4], np.linspace(0.0, 0.5, 3), n_samples=2,
                    save_path=npz, verbose=False, seed=0)

    meta_path = npz.replace('.npz', '_metadata.json')
    with open(meta_path) as f:
        meta = json.load(f)

    for field in REQUIRED_METADATA_FIELDS:
        assert field in meta, f"Missing required metadata field: '{field}'"


def test_metadata_values_consistent(tmp_path):
    """Metadata L_values, n_samples, and seed must match what was requested."""
    npz = str(tmp_path / 'out.npz')
    L_vals = [4, 6]
    n_samp = 3
    seed   = 77

    parameter_sweep(L_vals, np.linspace(0.0, 0.5, 4), n_samples=n_samp,
                    save_path=npz, verbose=False, seed=seed)

    with open(npz.replace('.npz', '_metadata.json')) as f:
        meta = json.load(f)

    assert meta['L_values'] == L_vals,    f"L_values mismatch: {meta['L_values']}"
    assert meta['n_samples'] == n_samp,   f"n_samples mismatch: {meta['n_samples']}"
    assert meta['random_seed'] == seed,   f"random_seed mismatch: {meta['random_seed']}"
    assert meta['measurements_applied_during_warmup'] is True
