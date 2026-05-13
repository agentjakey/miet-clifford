# Parameter sweeps and disorder averaging
import numpy as np
from tqdm import tqdm
from .stabilizer import StabilizerState
from .circuit import random_clifford_circuit
from .entropy import entanglement_entropy


def run_single(n, p_meas, n_layers, seed=None):
    """
    Evolve a single n-qubit state for n_layers layers and return S(L/2).
    """
    rng = np.random.default_rng(seed)
    state = StabilizerState(n)
    subsystem = list(range(n // 2))
    for _ in range(n_layers):
        random_clifford_circuit(state, p_meas, rng=rng)
    return entanglement_entropy(state, subsystem)


def run_sweep(L_values, p_values, n_layers, n_samples, seed=0, verbose=True):
    """
    Full (L, p) grid sweep with disorder averaging.
    Returns S_avg[i_L, i_p] averaged over n_samples disorder realizations.
    """
    rng = np.random.default_rng(seed)
    S_avg = np.zeros((len(L_values), len(p_values)))
    S_std = np.zeros_like(S_avg)
    iterator = tqdm(total=len(L_values) * len(p_values)) if verbose else None
    for i, L in enumerate(L_values):
        for j, p in enumerate(p_values):
            samples = [
                run_single(L, p, n_layers, seed=int(rng.integers(1 << 31)))
                for _ in range(n_samples)
            ]
            S_avg[i, j] = np.mean(samples)
            S_std[i, j] = np.std(samples)
            if iterator is not None:
                iterator.update(1)
    if iterator is not None:
        iterator.close()
    return S_avg, S_std
