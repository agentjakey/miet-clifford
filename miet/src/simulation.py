"""
Disorder-averaged simulation driver for the hybrid random Clifford circuit.
"""

import numpy as np
from tqdm import tqdm
from src.circuit import run_circuit
from src.entropy import half_chain_entropy
from src.metadata import collect_metadata, save_metadata


def single_sample(n_qubits, p_meas, n_steps, warmup=None, rng=None):
    state = run_circuit(n_qubits, p_meas, n_steps, warmup=warmup, rng=rng)
    return half_chain_entropy(state)


def disorder_average(n_qubits, p_meas, n_steps, n_samples, warmup=None, seed=None):
    """Run n_samples independent circuits. Returns dict with keys:
      'mean', 'std', 'sem', 'samples' (np.ndarray of shape (n_samples,))

    When seed is provided, each sample gets a deterministic child RNG derived
    from (seed, sample_index) via numpy SeedSequence. Results are fully
    reproducible: re-running with the same seed produces identical samples.
    When seed is None, uses the global numpy RNG (backward-compatible).
    """
    if seed is not None:
        ss = np.random.SeedSequence(seed)
        rngs = [np.random.default_rng(child) for child in ss.spawn(n_samples)]
    else:
        rngs = [None] * n_samples

    samples = np.array(
        [
            single_sample(n_qubits, p_meas, n_steps, warmup=warmup, rng=rng)
            for rng in rngs
        ]
    )
    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples, ddof=1)),
        "sem": float(np.std(samples, ddof=1) / np.sqrt(n_samples)),
        "samples": samples,
    }


def parameter_sweep(
    L_values,
    p_values,
    n_samples,
    n_steps_fn=None,
    warmup_fn=None,
    save_path=None,
    verbose=True,
    n_workers=1,
    seed=None,
):
    """Full (L, p) grid sweep with checkpoint saving after each L.

    Args:
        L_values:    list of system sizes
        p_values:    array of measurement rates
        n_samples:   disorder realizations per (L, p) point
        n_steps_fn:  callable (L) -> int, default 4*L
        warmup_fn:   callable (L) -> int, default 3*L
        save_path:   .npz path for checkpoints (None = no saving)
        verbose:     show tqdm progress bars
        n_workers:   reserved for future parallel use (currently ignored)
        seed:        integer master seed for reproducibility (None = non-deterministic)
    Returns:
        results[L][p] = dict with 'mean', 'std', 'sem', 'samples'
    """
    if n_steps_fn is None:
        n_steps_fn = lambda L: 4 * L
    if warmup_fn is None:
        warmup_fn = lambda L: 3 * L

    # Derive one child seed per (L, p) point from the master seed.
    # SeedSequence(None) gives a non-deterministic sequence (no fixed seed).
    ss = np.random.SeedSequence(seed)
    n_points = len(L_values) * len(p_values)
    child_seqs = ss.spawn(n_points)

    results = {}
    L_range = list(enumerate(L_values))
    L_iter = tqdm(L_range, desc="L sweep") if verbose else L_range

    for i, L in L_iter:
        results[L] = {}
        n_steps = n_steps_fn(L)
        warmup = warmup_fn(L)
        p_range = list(enumerate(p_values))
        p_iter = tqdm(p_range, desc=f"  L={L}", leave=False) if verbose else p_range

        for j, p in p_iter:
            # Unique, deterministic integer seed for this (L, p) point.
            point_seed = int(child_seqs[i * len(p_values) + j].generate_state(1)[0])
            results[L][float(p)] = disorder_average(
                L,
                float(p),
                n_steps,
                n_samples,
                warmup=warmup,
                seed=point_seed if seed is not None else None,
            )

        if save_path is not None:
            _save_results(results, L_values, p_values, save_path)
            if verbose:
                tqdm.write(f"Saved checkpoint after L={L}")

    # Write companion metadata JSON whenever we have a save path.
    if save_path is not None:
        meta = collect_metadata(
            L_values,
            p_values,
            n_samples,
            n_steps_fn,
            warmup_fn,
            seed,
            measurements_during_warmup=True,
        )
        meta_path = save_metadata(meta, save_path)
        if verbose:
            tqdm.write(f"Saved metadata  {meta_path}")

    return results


def _save_results(results, L_values, p_values, path):
    mean_arr = np.zeros((len(L_values), len(p_values)))
    std_arr = np.zeros_like(mean_arr)
    sem_arr = np.zeros_like(mean_arr)
    for i, L in enumerate(L_values):
        for j, p in enumerate(p_values):
            r = results.get(L, {}).get(float(p), {})
            mean_arr[i, j] = r.get("mean", np.nan)
            std_arr[i, j] = r.get("std", np.nan)
            sem_arr[i, j] = r.get("sem", np.nan)
    np.savez(
        path,
        L_values=np.array(L_values),
        p_values=p_values,
        mean_entropy=mean_arr,
        std_entropy=std_arr,
        sem_entropy=sem_arr,
    )


def load_results(path):
    """Returns (L_values list, p_values array, mean 2D, std 2D, sem 2D)."""
    d = np.load(path)
    return (
        list(d["L_values"].astype(int)),
        d["p_values"],
        d["mean_entropy"],
        d["std_entropy"],
        d["sem_entropy"],
    )
